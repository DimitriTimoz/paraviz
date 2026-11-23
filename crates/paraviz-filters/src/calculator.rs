//! Expression evaluation filter (`CalculatorFilter`, equivalent to `vtkArrayCalculator`)
//! with a recursive-descent parser.

use glam::Vec3;
use paraviz_core::{
    ArrayData, AttributeAssociation, DataArray, DataSet, FieldData, Filter, ImageData,
    MultiBlockDataSet, PipelineError, RectilinearGrid,
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
                let coords = dataset_point_coords(input);
                let fd = input.point_data().ok_or_else(|| {
                    PipelineError::ExecutionFailed("Dataset has no point_data".into())
                })?;
                let arr = eval_ast_on_field(&ast, &coords, fd, &self.result_name)?;
                out.point_data_mut().unwrap().add_array(arr);
            }
            AttributeAssociation::Cell => {
                let coords = dataset_cell_centroids(input);
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

enum CoordSource<'a> {
    Explicit(&'a [[f32; 3]]),
    ImagePoints(&'a ImageData),
    RectilinearPoints(&'a RectilinearGrid),
    ImageCellCentroids(&'a ImageData),
    RectilinearCellCentroids(&'a RectilinearGrid),
    Owned(Vec<[f32; 3]>),
}

impl<'a> CoordSource<'a> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Explicit(pts) => pts.len(),
            Self::ImagePoints(img) => img.num_points(),
            Self::RectilinearPoints(rg) => rg.num_points(),
            Self::ImageCellCentroids(img) => img.num_cells(),
            Self::RectilinearCellCentroids(rg) => {
                let [nx, ny, nz] = rg.dimensions();
                if nx == 0 || ny == 0 || nz == 0 {
                    0
                } else {
                    rg.num_cells()
                }
            }
            Self::Owned(pts) => pts.len(),
        }
    }

    #[inline]
    fn at(&self, idx: usize) -> [f32; 3] {
        match self {
            Self::Explicit(pts) => pts[idx],
            Self::ImagePoints(img) => {
                let [nx, ny, _] = img.dimensions;
                let i = idx % nx;
                let rem = idx / nx;
                let j = rem % ny;
                let k = rem / ny;
                img.point_coords(i, j, k)
            }
            Self::RectilinearPoints(rg) => {
                let [nx, ny, _] = rg.dimensions();
                let i = idx % nx;
                let rem = idx / nx;
                let j = rem % ny;
                let k = rem / ny;
                rg.point_coords(i, j, k)
            }
            Self::ImageCellCentroids(img) => {
                let [nx, ny, nz] = img.dimensions;
                let cx = nx.saturating_sub(1).max(1);
                let cy = ny.saturating_sub(1).max(1);
                let i = idx % cx;
                let rem = idx / cx;
                let j = rem % cy;
                let k = rem / cy;
                let x = img.origin[0] + (i as f64 + if nx > 1 { 0.5 } else { 0.0 }) * img.spacing[0];
                let y = img.origin[1] + (j as f64 + if ny > 1 { 0.5 } else { 0.0 }) * img.spacing[1];
                let z = img.origin[2] + (k as f64 + if nz > 1 { 0.5 } else { 0.0 }) * img.spacing[2];
                [x as f32, y as f32, z as f32]
            }
            Self::RectilinearCellCentroids(rg) => {
                let [nx, ny, nz] = rg.dimensions();
                let cx = nx.saturating_sub(1).max(1);
                let cy = ny.saturating_sub(1).max(1);
                let i = idx % cx;
                let rem = idx / cx;
                let j = rem % cy;
                let k = rem / cy;
                let x = if nx > 1 { 0.5 * (rg.x_coords[i] + rg.x_coords[i + 1]) } else { rg.x_coords[0] };
                let y = if ny > 1 { 0.5 * (rg.y_coords[j] + rg.y_coords[j + 1]) } else { rg.y_coords[0] };
                let z = if nz > 1 { 0.5 * (rg.z_coords[k] + rg.z_coords[k + 1]) } else { rg.z_coords[0] };
                [x, y, z]
            }
            Self::Owned(pts) => pts[idx],
        }
    }
}

fn dataset_point_coords(ds: &DataSet) -> CoordSource<'_> {
    match ds {
        DataSet::PolyData(pd) => CoordSource::Explicit(&pd.points),
        DataSet::ImageData(img) => CoordSource::ImagePoints(img),
        DataSet::RectilinearGrid(rg) => CoordSource::RectilinearPoints(rg),
        DataSet::StructuredGrid(sg) => CoordSource::Explicit(&sg.points),
        DataSet::UnstructuredGrid(ug) => CoordSource::Explicit(&ug.points),
        DataSet::MultiBlock(_) => unreachable!(),
    }
}

fn dataset_cell_centroids(ds: &DataSet) -> CoordSource<'_> {
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
            CoordSource::Owned(c)
        }
        DataSet::ImageData(img) => CoordSource::ImageCellCentroids(img),
        DataSet::RectilinearGrid(rg) => CoordSource::RectilinearCellCentroids(rg),
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
            CoordSource::Owned(out)
        }
        DataSet::UnstructuredGrid(ug) => CoordSource::Owned(
            (0..ug.num_cells())
                .map(|c| avg_pts(&ug.points, ug.cell_points(c)))
                .collect(),
        ),
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
    coords: &CoordSource<'_>,
    fd: &FieldData,
    result_name: &str,
) -> Result<DataArray, PipelineError> {
    let n = coords.len();
    if n == 0 {
        return Ok(DataArray::scalars_f32(result_name, Vec::new()));
    }
    let bound = bind_expr(ast, fd, n)?;
    // Probe sample 0 to determine scalar vs vector output.
    let first = eval_bound(&bound, 0, coords);
    match first {
        Val::Scalar(_) => {
            let vals: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| eval_bound(&bound, i, coords).as_scalar() as f32)
                .collect();
            Ok(DataArray::scalars_f32(result_name, vals))
        }
        Val::Vec3(_) => {
            let vals: Vec<[f32; 3]> = (0..n)
                .into_par_iter()
                .map(|i| match eval_bound(&bound, i, coords) {
                    Val::Vec3(v) => [v[0] as f32, v[1] as f32, v[2] as f32],
                    Val::Scalar(v) => [v as f32, v as f32, v as f32],
                })
                .collect();
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

#[derive(Clone, Copy)]
enum BoundSlice<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl<'a> BoundSlice<'a> {
    #[inline]
    fn from_array_data(data: &'a ArrayData) -> Self {
        match data {
            ArrayData::F32(v) => Self::F32(v),
            ArrayData::F64(v) => Self::F64(v),
            ArrayData::I8(v) => Self::I8(v),
            ArrayData::I16(v) => Self::I16(v),
            ArrayData::I32(v) => Self::I32(v),
            ArrayData::I64(v) => Self::I64(v),
            ArrayData::U8(v) => Self::U8(v),
            ArrayData::U16(v) => Self::U16(v),
            ArrayData::U32(v) => Self::U32(v),
            ArrayData::U64(v) => Self::U64(v),
        }
    }

    #[inline]
    fn get_f64(self, idx: usize) -> f64 {
        match self {
            Self::F32(v) => v[idx] as f64,
            Self::F64(v) => v[idx],
            Self::I8(v) => v[idx] as f64,
            Self::I16(v) => v[idx] as f64,
            Self::I32(v) => v[idx] as f64,
            Self::I64(v) => v[idx] as f64,
            Self::U8(v) => v[idx] as f64,
            Self::U16(v) => v[idx] as f64,
            Self::U32(v) => v[idx] as f64,
            Self::U64(v) => v[idx] as f64,
        }
    }
}

#[derive(Clone, Copy)]
enum BuiltinFn {
    Mag,
    Sin,
    Cos,
    Tan,
    Sqrt,
    Abs,
    Exp,
    Ln,
    Min,
    Max,
    Clamp,
}

enum BoundExpr<'a> {
    Num(f64),
    CoordX,
    CoordY,
    CoordZ,
    Coords,
    ScalarArray { slice: BoundSlice<'a>, nc: usize },
    Vec3Array { slice: BoundSlice<'a>, nc: usize },
    Neg(Box<BoundExpr<'a>>),
    Bin(BinOp, Box<BoundExpr<'a>>, Box<BoundExpr<'a>>),
    Call(BuiltinFn, Vec<BoundExpr<'a>>),
}

fn bind_expr<'a>(
    expr: &Expr,
    fd: &'a FieldData,
    expected_len: usize,
) -> Result<BoundExpr<'a>, PipelineError> {
    match expr {
        Expr::Num(v) => Ok(BoundExpr::Num(*v)),
        Expr::Ident(name) => match name.as_str() {
            "coordsX" => Ok(BoundExpr::CoordX),
            "coordsY" => Ok(BoundExpr::CoordY),
            "coordsZ" => Ok(BoundExpr::CoordZ),
            "coords" => Ok(BoundExpr::Coords),
            _ => {
                if let Some(arr) = fd.get(name) {
                    if expected_len > arr.num_tuples() {
                        return Err(PipelineError::ExecutionFailed(format!(
                            "Array '{name}' length ({}) does not match dataset size",
                            arr.num_tuples()
                        )));
                    }
                    let slice = BoundSlice::from_array_data(&arr.data);
                    let nc = arr.num_components;
                    if nc >= 3 {
                        Ok(BoundExpr::Vec3Array { slice, nc })
                    } else {
                        Ok(BoundExpr::ScalarArray { slice, nc })
                    }
                } else {
                    match name.as_str() {
                        "pi" | "PI" => Ok(BoundExpr::Num(std::f64::consts::PI)),
                        "e" | "E" => Ok(BoundExpr::Num(std::f64::consts::E)),
                        _ => Err(PipelineError::ExecutionFailed(format!(
                            "Unknown variable or array '{name}' in CalculatorFilter"
                        ))),
                    }
                }
            }
        },
        Expr::Neg(inner) => Ok(BoundExpr::Neg(Box::new(bind_expr(inner, fd, expected_len)?))),
        Expr::Bin(op, a, b) => Ok(BoundExpr::Bin(
            *op,
            Box::new(bind_expr(a, fd, expected_len)?),
            Box::new(bind_expr(b, fd, expected_len)?),
        )),
        Expr::Call(fn_name, args) => {
            let (builtin, expected_arity) = match fn_name.as_str() {
                "mag" => (BuiltinFn::Mag, 1),
                "sin" => (BuiltinFn::Sin, 1),
                "cos" => (BuiltinFn::Cos, 1),
                "tan" => (BuiltinFn::Tan, 1),
                "sqrt" => (BuiltinFn::Sqrt, 1),
                "abs" => (BuiltinFn::Abs, 1),
                "exp" => (BuiltinFn::Exp, 1),
                "ln" => (BuiltinFn::Ln, 1),
                "min" => (BuiltinFn::Min, 2),
                "max" => (BuiltinFn::Max, 2),
                "clamp" => (BuiltinFn::Clamp, 3),
                _ => {
                    return Err(PipelineError::ExecutionFailed(format!(
                        "Unknown function '{fn_name}' in CalculatorFilter"
                    )));
                }
            };
            check_arity(fn_name, args, expected_arity)?;
            let bound_args = args
                .iter()
                .map(|a| bind_expr(a, fd, expected_len))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(BoundExpr::Call(builtin, bound_args))
        }
    }
}

#[inline]
fn eval_bound(expr: &BoundExpr<'_>, idx: usize, coords: &CoordSource<'_>) -> Val {
    match expr {
        BoundExpr::Num(v) => Val::Scalar(*v),
        BoundExpr::CoordX => Val::Scalar(coords.at(idx)[0] as f64),
        BoundExpr::CoordY => Val::Scalar(coords.at(idx)[1] as f64),
        BoundExpr::CoordZ => Val::Scalar(coords.at(idx)[2] as f64),
        BoundExpr::Coords => {
            let c = coords.at(idx);
            Val::Vec3([c[0] as f64, c[1] as f64, c[2] as f64])
        }
        BoundExpr::ScalarArray { slice, nc } => Val::Scalar(slice.get_f64(idx * *nc)),
        BoundExpr::Vec3Array { slice, nc } => {
            let base = idx * *nc;
            Val::Vec3([
                slice.get_f64(base),
                slice.get_f64(base + 1),
                slice.get_f64(base + 2),
            ])
        }
        BoundExpr::Neg(inner) => match eval_bound(inner, idx, coords) {
            Val::Scalar(s) => Val::Scalar(-s),
            Val::Vec3(v) => Val::Vec3([-v[0], -v[1], -v[2]]),
        },
        BoundExpr::Bin(op, a, b) => {
            let va = eval_bound(a, idx, coords);
            let vb = eval_bound(b, idx, coords);
            match (op, va, vb) {
                (BinOp::Add, Val::Vec3(u), Val::Vec3(v)) => {
                    Val::Vec3([u[0] + v[0], u[1] + v[1], u[2] + v[2]])
                }
                (BinOp::Sub, Val::Vec3(u), Val::Vec3(v)) => {
                    Val::Vec3([u[0] - v[0], u[1] - v[1], u[2] - v[2]])
                }
                (BinOp::Mul, Val::Vec3(u), Val::Scalar(s))
                | (BinOp::Mul, Val::Scalar(s), Val::Vec3(u)) => {
                    Val::Vec3([u[0] * s, u[1] * s, u[2] * s])
                }
                (BinOp::Div, Val::Vec3(u), Val::Scalar(s)) => {
                    Val::Vec3([u[0] / s, u[1] / s, u[2] / s])
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
                    Val::Scalar(r)
                }
            }
        }
        BoundExpr::Call(builtin, args) => match builtin {
            BuiltinFn::Mag => Val::Scalar(eval_bound(&args[0], idx, coords).mag()),
            BuiltinFn::Sin => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().sin()),
            BuiltinFn::Cos => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().cos()),
            BuiltinFn::Tan => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().tan()),
            BuiltinFn::Sqrt => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().sqrt()),
            BuiltinFn::Abs => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().abs()),
            BuiltinFn::Exp => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().exp()),
            BuiltinFn::Ln => Val::Scalar(eval_bound(&args[0], idx, coords).as_scalar().ln()),
            BuiltinFn::Min => {
                let a = eval_bound(&args[0], idx, coords).as_scalar();
                let b = eval_bound(&args[1], idx, coords).as_scalar();
                Val::Scalar(a.min(b))
            }
            BuiltinFn::Max => {
                let a = eval_bound(&args[0], idx, coords).as_scalar();
                let b = eval_bound(&args[1], idx, coords).as_scalar();
                Val::Scalar(a.max(b))
            }
            BuiltinFn::Clamp => {
                let x = eval_bound(&args[0], idx, coords).as_scalar();
                let lo = eval_bound(&args[1], idx, coords).as_scalar();
                let hi = eval_bound(&args[2], idx, coords).as_scalar();
                Val::Scalar(x.clamp(lo, hi))
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_evaluates_rectilinear_coords_and_arrays_without_all_points() {
        let mut rg = RectilinearGrid {
            x_coords: vec![1.0, 3.0],
            y_coords: vec![2.0, 4.0],
            z_coords: vec![0.0, 5.0],
            ..Default::default()
        };
        let n = rg.num_points();
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        rg.point_data.set_scalars(DataArray::scalars_f32("p", vals));

        let filter = CalculatorFilter::new(
            "mag(coords) + p",
            "out",
            AttributeAssociation::Point,
        );
        let out = filter.apply(&DataSet::RectilinearGrid(rg.clone())).unwrap();
        let out_arr = out.point_data().unwrap().get("out").unwrap();
        assert_eq!(out_arr.num_tuples(), n);
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    let idx = rg.point_index(i, j, k);
                    let c = rg.point_coords(i, j, k);
                    let expected = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() + idx as f32;
                    assert!((out_arr.get_scalar(idx) as f32 - expected).abs() < 1e-5);
                }
            }
        }
    }
}
