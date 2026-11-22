//! Expression evaluation filter (`CalculatorFilter`, equivalent to `vtkArrayCalculator`)
//! with a recursive-descent parser.

use glam::Vec3;
use paraviz_core::{
    AttributeAssociation, DataArray, DataSet, FieldData, Filter, MultiBlockDataSet, PipelineError,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Evaluates a mathematical expression across all points or cells of a dataset,
/// storing the result as a new `DataArray`.
///
/// Supported syntax:
/// - Arithmetic: `+`, `-`, `*`, `/`, `^` (power), unary `-`/`+`, parentheses `(...)`
/// - Coordinates: `coordsX`, `coordsY`, `coordsZ` (and `coords` vector)
/// - Functions: `mag(v)`, `sin(x)`, `cos(x)`, `tan(x)`, `sqrt(x)`, `abs(x)`, `exp(x)`, `ln(x)`,
///   `min(a,b)`, `max(a,b)`, `clamp(x,lo,hi)`
/// - Any point/cell array name in the dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalculatorFilter {
    pub expression: String,
    pub result_name: String,
    pub association: AttributeAssociation,
}

impl Default for CalculatorFilter {
    fn default() -> Self {
        Self {
            expression: "coordsX".into(),
            result_name: "Result".into(),
            association: AttributeAssociation::Point,
        }
    }
}

impl CalculatorFilter {
    pub fn new(
        expression: impl Into<String>,
        result_name: impl Into<String>,
        association: AttributeAssociation,
    ) -> Self {
        Self {
            expression: expression.into(),
            result_name: result_name.into(),
            association,
        }
    }
}

impl Filter for CalculatorFilter {
    fn name(&self) -> &str {
        "CalculatorFilter"
    }

    fn apply(&self, input: &DataSet) -> Result<DataSet, PipelineError> {
        if let DataSet::MultiBlock(mb) = input {
            let mut out = MultiBlockDataSet::new();
            for (name, block) in &mb.blocks {
                out.add_block(name.clone(), self.apply(block)?);
            }
            return Ok(DataSet::MultiBlock(out));
        }

        let ast = parse_expression(&self.expression)?;
        let mut out = input.clone();

        match self.association {
            AttributeAssociation::Point => {
                let coords = dataset_point_coords(input)?;
                let fd = input.point_data().ok_or_else(|| {
                    PipelineError::ExecutionFailed("Dataset has no point_data".into())
                })?;
                let arr = eval_ast_on_field(&ast, &coords, fd, &self.result_name)?;
                out.point_data_mut().unwrap().add_array(arr);
            }
            AttributeAssociation::Cell => {
                let coords = dataset_cell_centroids(input)?;
                let fd = input.cell_data().ok_or_else(|| {
                    PipelineError::ExecutionFailed("Dataset has no cell_data".into())
                })?;
                let arr = eval_ast_on_field(&ast, &coords, fd, &self.result_name)?;
                out.cell_data_mut().unwrap().add_array(arr);
            }
            AttributeAssociation::Field => {
                return Err(PipelineError::ExecutionFailed(
                    "CalculatorFilter requires Point or Cell association".into(),
                ));
            }
        }

        Ok(out)
    }
}

fn dataset_point_coords(ds: &DataSet) -> Result<Vec<[f32; 3]>, PipelineError> {
    match ds {
        DataSet::PolyData(pd) => Ok(pd.points.clone()),
        DataSet::ImageData(img) => Ok(img.all_points()),
        DataSet::RectilinearGrid(rg) => Ok(rg.all_points()),
        DataSet::StructuredGrid(sg) => Ok(sg.points.clone()),
        DataSet::UnstructuredGrid(ug) => Ok(ug.points.clone()),
        DataSet::MultiBlock(_) => unreachable!(),
    }
}

fn dataset_cell_centroids(ds: &DataSet) -> Result<Vec<[f32; 3]>, PipelineError> {
    match ds {
        DataSet::PolyData(pd) => {
            let mut c = Vec::with_capacity(pd.num_cells());
            for &v in &pd.verts {
                c.push(pd.points[v as usize]);
            }
            for line in &pd.lines {
                c.push(avg_pts(&pd.points, line));
            }
            for poly in &pd.polys {
                c.push(avg_pts(&pd.points, poly));
            }
            for strip in &pd.strips {
                c.push(avg_pts(&pd.points, strip));
            }
            Ok(c)
        }
        DataSet::ImageData(img) => {
            let [nx, ny, nz] = img.dimensions;
            let cx = nx.saturating_sub(1).max(1);
            let cy = ny.saturating_sub(1).max(1);
            let cz = nz.saturating_sub(1).max(1);
            let mut out = Vec::with_capacity(img.num_cells());
            for k in 0..cz {
                for j in 0..cy {
                    for i in 0..cx {
                        let x = img.origin[0] + (i as f64 + if nx > 1 { 0.5 } else { 0.0 }) * img.spacing[0];
                        let y = img.origin[1] + (j as f64 + if ny > 1 { 0.5 } else { 0.0 }) * img.spacing[1];
                        let z = img.origin[2] + (k as f64 + if nz > 1 { 0.5 } else { 0.0 }) * img.spacing[2];
                        out.push([x as f32, y as f32, z as f32]);
                    }
                }
            }
            Ok(out)
        }
        DataSet::RectilinearGrid(rg) => {
            let [nx, ny, nz] = rg.dimensions();
            let cx = nx.saturating_sub(1).max(1);
            let cy = ny.saturating_sub(1).max(1);
            let cz = nz.saturating_sub(1).max(1);
            let mut out = Vec::with_capacity(cx * cy * cz);
            for k in 0..cz {
                let z = if nz > 1 { 0.5 * (rg.z_coords[k] + rg.z_coords[k + 1]) } else { rg.z_coords[0] };
                for j in 0..cy {
                    let y = if ny > 1 { 0.5 * (rg.y_coords[j] + rg.y_coords[j + 1]) } else { rg.y_coords[0] };
                    for i in 0..cx {
                        let x = if nx > 1 { 0.5 * (rg.x_coords[i] + rg.x_coords[i + 1]) } else { rg.x_coords[0] };
                        out.push([x, y, z]);
                    }
                }
            }
            Ok(out)
        }
        DataSet::StructuredGrid(sg) => {
            let [nx, ny, nz] = sg.dimensions;
            let cx = nx.saturating_sub(1).max(1);
            let cy = ny.saturating_sub(1).max(1);
            let cz = nz.saturating_sub(1).max(1);
            let mut out = Vec::with_capacity(cx * cy * cz);
            for k in 0..cz {
                for j in 0..cy {
                    for i in 0..cx {
                        let mut sum = Vec3::ZERO;
                        let mut cnt = 0.0f32;
                        for dk in 0..=usize::from(nz > 1) {
                            for dj in 0..=usize::from(ny > 1) {
                                for di in 0..=usize::from(nx > 1) {
                                    sum += Vec3::from_array(sg.point_coords(i + di, j + dj, k + dk));
                                    cnt += 1.0;
                                }
                            }
                        }
                        out.push((sum / cnt).to_array());
                    }
                }
            }
            Ok(out)
        }
        DataSet::UnstructuredGrid(ug) => Ok((0..ug.num_cells())
            .map(|c| avg_pts(&ug.points, ug.cell_points(c)))
            .collect()),
        DataSet::MultiBlock(_) => unreachable!(),
    }
}

fn avg_pts(pts: &[[f32; 3]], indices: &[u32]) -> [f32; 3] {
    if indices.is_empty() {
        return [0.0; 3];
    }
    let mut sum = Vec3::ZERO;
    for &i in indices {
        sum += Vec3::from_array(pts[i as usize]);
    }
    (sum / indices.len() as f32).to_array()
}

fn eval_ast_on_field(
    ast: &Expr,
    coords: &[[f32; 3]],
    fd: &FieldData,
    result_name: &str,
) -> Result<DataArray, PipelineError> {
    let n = coords.len();
    if n == 0 {
        return Ok(DataArray::scalars_f32(result_name, Vec::new()));
    }
    // Probe sample 0 to validate variables and determine scalar vs vector output.
    let first = eval_expr(ast, 0, coords[0], fd)?;
    match first {
        Val::Scalar(_) => {
            let vals: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| match eval_expr(ast, i, coords[i], fd) {
                    Ok(Val::Scalar(v)) => Ok(v as f32),
                    Ok(Val::Vec3(v)) => Ok(v[0] as f32),
                    Err(e) => Err(e),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DataArray::scalars_f32(result_name, vals))
        }
        Val::Vec3(_) => {
            let vals: Vec<[f32; 3]> = (0..n)
                .into_par_iter()
                .map(|i| match eval_expr(ast, i, coords[i], fd) {
                    Ok(Val::Vec3(v)) => Ok([v[0] as f32, v[1] as f32, v[2] as f32]),
                    Ok(Val::Scalar(v)) => Ok([v as f32, v as f32, v as f32]),
                    Err(e) => Err(e),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DataArray::vectors_f32(result_name, vals))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Val {
    Scalar(f64),
    Vec3([f64; 3]),
}

impl Val {
    fn as_scalar(self) -> f64 {
        match self {
            Self::Scalar(s) => s,
            Self::Vec3(v) => v[0],
        }
    }

    fn mag(self) -> f64 {
        match self {
            Self::Scalar(s) => s.abs(),
            Self::Vec3(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Ident(String),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, PipelineError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        match ch {
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.')
                {
                    i += 1;
                }
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
                let s = &input[start..i];
                let num = s.parse::<f64>().map_err(|_| {
                    PipelineError::ExecutionFailed(format!("Invalid number literal '{s}'"))
                })?;
                tokens.push(Token::Num(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < bytes.len() {
                    let b = bytes[i] as char;
                    if b.is_ascii_alphanumeric() || b == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(input[start..i].to_string()));
            }
            other => {
                return Err(PipelineError::ExecutionFailed(format!(
                    "Unexpected character '{other}' in CalculatorFilter expression"
                )));
            }
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expr(&mut self) -> Result<Expr, PipelineError> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, PipelineError> {
        let mut left = self.parse_mul_div()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.next();
                    let right = self.parse_mul_div()?;
                    left = Expr::Bin(BinOp::Add, Box::new(left), Box::new(right));
                }
                Some(Token::Minus) => {
                    self.next();
                    let right = self.parse_mul_div()?;
                    left = Expr::Bin(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, PipelineError> {
        let mut left = self.parse_pow()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.next();
                    let right = self.parse_pow()?;
                    left = Expr::Bin(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Some(Token::Slash) => {
                    self.next();
                    let right = self.parse_pow()?;
                    left = Expr::Bin(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_pow(&mut self) -> Result<Expr, PipelineError> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.next();
            let exp = self.parse_pow()?; // right-associative
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, PipelineError> {
        match self.peek() {
            Some(Token::Minus) => {
                self.next();
                Ok(Expr::Neg(Box::new(self.parse_unary()?)))
            }
            Some(Token::Plus) => {
                self.next();
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, PipelineError> {
        match self.next() {
            Some(Token::Num(v)) => Ok(Expr::Num(v)),
            Some(Token::Ident(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Token::Comma)) {
                                self.next();
                            } else {
                                break;
                            }
                        }
                    }
                    if !matches!(self.next(), Some(Token::RParen)) {
                        return Err(PipelineError::ExecutionFailed(format!(
                            "Expected ')' after arguments to '{name}'"
                        )));
                    }
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                if !matches!(self.next(), Some(Token::RParen)) {
                    return Err(PipelineError::ExecutionFailed(
                        "Mismatched '(' in CalculatorFilter expression".into(),
                    ));
                }
                Ok(inner)
            }
            other => Err(PipelineError::ExecutionFailed(format!(
                "Unexpected token {other:?} in CalculatorFilter expression"
            ))),
        }
    }
}

fn parse_expression(expr: &str) -> Result<Expr, PipelineError> {
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        return Err(PipelineError::ExecutionFailed(
            "CalculatorFilter expression is empty".into(),
        ));
    }
    let mut parser = Parser { tokens, pos: 0 };
    let ast = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        return Err(PipelineError::ExecutionFailed(format!(
            "Unexpected trailing token {:?} in expression",
            parser.tokens[parser.pos]
        )));
    }
    Ok(ast)
}

fn eval_expr(expr: &Expr, idx: usize, coord: [f32; 3], fd: &FieldData) -> Result<Val, PipelineError> {
    match expr {
        Expr::Num(v) => Ok(Val::Scalar(*v)),
        Expr::Ident(name) => match name.as_str() {
            "coordsX" => Ok(Val::Scalar(coord[0] as f64)),
            "coordsY" => Ok(Val::Scalar(coord[1] as f64)),
            "coordsZ" => Ok(Val::Scalar(coord[2] as f64)),
            "coords" => Ok(Val::Vec3([coord[0] as f64, coord[1] as f64, coord[2] as f64])),
            _ => {
                if let Some(arr) = fd.get(name) {
                    if idx >= arr.num_tuples() {
                        return Err(PipelineError::ExecutionFailed(format!(
                            "Array '{name}' length ({}) does not match dataset size",
                            arr.num_tuples()
                        )));
                    }
                    if arr.num_components >= 3 {
                        let base = idx * arr.num_components;
                        Ok(Val::Vec3([
                            arr.data.get_f64(base),
                            arr.data.get_f64(base + 1),
                            arr.data.get_f64(base + 2),
                        ]))
                    } else {
                        Ok(Val::Scalar(arr.get_scalar(idx)))
                    }
                } else {
                    match name.as_str() {
                        "pi" | "PI" => Ok(Val::Scalar(std::f64::consts::PI)),
                        "e" | "E" => Ok(Val::Scalar(std::f64::consts::E)),
                        _ => Err(PipelineError::ExecutionFailed(format!(
                            "Unknown variable or array '{name}' in CalculatorFilter"
                        ))),
                    }
                }
            }
        },
        Expr::Neg(inner) => match eval_expr(inner, idx, coord, fd)? {
            Val::Scalar(s) => Ok(Val::Scalar(-s)),
            Val::Vec3(v) => Ok(Val::Vec3([-v[0], -v[1], -v[2]])),
        },
        Expr::Bin(op, a, b) => {
            let va = eval_expr(a, idx, coord, fd)?;
            let vb = eval_expr(b, idx, coord, fd)?;
            match (op, va, vb) {
                (BinOp::Add, Val::Vec3(u), Val::Vec3(v)) => {
                    Ok(Val::Vec3([u[0] + v[0], u[1] + v[1], u[2] + v[2]]))
                }
                (BinOp::Sub, Val::Vec3(u), Val::Vec3(v)) => {
                    Ok(Val::Vec3([u[0] - v[0], u[1] - v[1], u[2] - v[2]]))
                }
                (BinOp::Mul, Val::Vec3(u), Val::Scalar(s))
                | (BinOp::Mul, Val::Scalar(s), Val::Vec3(u)) => {
                    Ok(Val::Vec3([u[0] * s, u[1] * s, u[2] * s]))
                }
                (BinOp::Div, Val::Vec3(u), Val::Scalar(s)) => {
                    Ok(Val::Vec3([u[0] / s, u[1] / s, u[2] / s]))
                }
                _ => {
                    let sa = va.as_scalar();
                    let sb = vb.as_scalar();
                    let r = match op {
                        BinOp::Add => sa + sb,
                        BinOp::Sub => sa - sb,
                        BinOp::Mul => sa * sb,
                        BinOp::Div => sa / sb,
                        BinOp::Pow => sa.powf(sb),
                    };
                    Ok(Val::Scalar(r))
                }
            }
        }
        Expr::Call(fn_name, args) => match fn_name.as_str() {
            "mag" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.mag()))
            }
            "sin" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().sin()))
            }
            "cos" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().cos()))
            }
            "tan" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().tan()))
            }
            "sqrt" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().sqrt()))
            }
            "abs" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().abs()))
            }
            "exp" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().exp()))
            }
            "ln" => {
                check_arity(fn_name, args, 1)?;
                Ok(Val::Scalar(eval_expr(&args[0], idx, coord, fd)?.as_scalar().ln()))
            }
            "min" => {
                check_arity(fn_name, args, 2)?;
                let a = eval_expr(&args[0], idx, coord, fd)?.as_scalar();
                let b = eval_expr(&args[1], idx, coord, fd)?.as_scalar();
                Ok(Val::Scalar(a.min(b)))
            }
            "max" => {
                check_arity(fn_name, args, 2)?;
                let a = eval_expr(&args[0], idx, coord, fd)?.as_scalar();
                let b = eval_expr(&args[1], idx, coord, fd)?.as_scalar();
                Ok(Val::Scalar(a.max(b)))
            }
            "clamp" => {
                check_arity(fn_name, args, 3)?;
                let x = eval_expr(&args[0], idx, coord, fd)?.as_scalar();
                let lo = eval_expr(&args[1], idx, coord, fd)?.as_scalar();
                let hi = eval_expr(&args[2], idx, coord, fd)?.as_scalar();
                Ok(Val::Scalar(x.clamp(lo, hi)))
            }
            _ => Err(PipelineError::ExecutionFailed(format!(
                "Unknown function '{fn_name}' in CalculatorFilter"
            ))),
        },
    }
}

fn check_arity(name: &str, args: &[Expr], expected: usize) -> Result<(), PipelineError> {
    if args.len() != expected {
        return Err(PipelineError::ExecutionFailed(format!(
            "Function '{name}' expects {expected} argument(s), got {}",
            args.len()
        )));
    }
    Ok(())
}
