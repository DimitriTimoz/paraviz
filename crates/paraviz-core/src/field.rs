//! PointData, CellData, and FieldData attribute containers (equivalent to `vtkFieldData` / `vtkDataSetAttributes`).

use crate::array::DataArray;
use serde::{Deserialize, Serialize};

/// Where an attribute array is attached on a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttributeAssociation {
    Point,
    Cell,
    Field,
}

/// Collection of named [`DataArray`]s with optional active scalar/vector/normal designations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FieldData {
    arrays: Vec<DataArray>,
    pub active_scalars: Option<String>,
    pub active_vectors: Option<String>,
    pub active_normals: Option<String>,
}

impl FieldData {
    /// Creates an empty [`FieldData`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces an array with the same name.
    pub fn add_array(&mut self, array: DataArray) {
        if let Some(existing) = self.arrays.iter_mut().find(|a| a.name == array.name) {
            *existing = array;
        } else {
            self.arrays.push(array);
        }
    }

    /// Adds a scalar array and marks it as `active_scalars`.
    pub fn set_scalars(&mut self, array: DataArray) {
        self.active_scalars = Some(array.name.clone());
        self.add_array(array);
    }

    /// Adds a vector array and marks it as `active_vectors`.
    pub fn set_vectors(&mut self, array: DataArray) {
        self.active_vectors = Some(array.name.clone());
        self.add_array(array);
    }

    /// Adds a normal array and marks it as `active_normals`.
    pub fn set_normals(&mut self, array: DataArray) {
        self.active_normals = Some(array.name.clone());
        self.add_array(array);
    }

    /// Gets an array by name (accepts `&str`, `String`, `&String`, ...).
    pub fn get(&self, name: impl AsRef<str>) -> Option<&DataArray> {
        let name = name.as_ref();
        self.arrays.iter().find(|a| a.name == name)
    }

    /// Gets a mutable reference to an array by name (accepts `&str`, `String`, ...).
    pub fn get_mut(&mut self, name: impl AsRef<str>) -> Option<&mut DataArray> {
        let name = name.as_ref();
        self.arrays.iter_mut().find(|a| a.name == name)
    }

    /// Removes an array by name, returning it if it existed.
    pub fn remove(&mut self, name: impl AsRef<str>) -> Option<DataArray> {
        let name = name.as_ref();
        if let Some(pos) = self.arrays.iter().position(|a| a.name == name) {
            if self.active_scalars.as_deref() == Some(name) {
                self.active_scalars = None;
            }
            if self.active_vectors.as_deref() == Some(name) {
                self.active_vectors = None;
            }
            if self.active_normals.as_deref() == Some(name) {
                self.active_normals = None;
            }
            Some(self.arrays.remove(pos))
        } else {
            None
        }
    }

    /// Returns all stored arrays.
    pub fn arrays(&self) -> &[DataArray] {
        &self.arrays
    }

    /// Returns mutable slice of all stored arrays.
    pub fn arrays_mut(&mut self) -> &mut [DataArray] {
        &mut self.arrays
    }

    /// Lists array names in insertion order.
    pub fn array_names(&self) -> Vec<&str> {
        self.arrays.iter().map(|a| a.name.as_str()).collect()
    }

    /// Returns the active scalar array, or the first 1-component array if none explicitly set.
    pub fn active_scalars_array(&self) -> Option<&DataArray> {
        if let Some(name) = &self.active_scalars
            && let Some(arr) = self.get(name)
        {
            return Some(arr);
        }
        self.arrays.iter().find(|a| a.num_components == 1)
    }

    /// Returns the active vector array, or the first 3-component non-normal array if none explicitly set.
    pub fn active_vectors_array(&self) -> Option<&DataArray> {
        if let Some(name) = &self.active_vectors
            && let Some(arr) = self.get(name)
        {
            return Some(arr);
        }
        self.arrays
            .iter()
            .find(|a| a.num_components == 3 && self.active_normals.as_deref() != Some(&a.name))
    }

    /// Returns the active normals array if present.
    pub fn active_normals_array(&self) -> Option<&DataArray> {
        if let Some(name) = &self.active_normals
            && let Some(arr) = self.get(name)
        {
            return Some(arr);
        }
        self.get("Normals").or_else(|| self.get("normals"))
    }

    /// Gathers a subset of tuple indices across all arrays, preserving active flags.
    pub fn gather(&self, indices: &[usize]) -> Self {
        Self {
            arrays: self.arrays.iter().map(|a| a.gather(indices)).collect(),
            active_scalars: self.active_scalars.clone(),
            active_vectors: self.active_vectors.clone(),
            active_normals: self.active_normals.clone(),
        }
    }

    /// Returns `true` if no arrays are stored.
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Number of stored arrays.
    pub fn len(&self) -> usize {
        self.arrays.len()
    }
}
