use ndarray::*;
use std::hash::{Hash, Hasher};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::core::variable::{VarData, Variable};

/// Sum elements along axes to output an array of a given shape.
///
/// # Arguments
/// * `x` - Input array.
/// * `shape` - Desired output shape.
///
/// # Returns
/// * `Array<f64, IxDyn>` - Output array of the specified shape.
pub fn sum_to(x: &Array<f64, IxDyn>, shape: &[usize]) -> Array<f64, IxDyn> {
    let ndim = shape.len();
    let lead = x.ndim() - ndim;
    let lead_axes: Vec<usize> = (0..lead).collect();

    // Collect axes where the desired shape has size 1
    let axis: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &sx)| if sx == 1 { Some(i + lead) } else { None })
        .collect();

    // Sum over all specified axes, keeping dimensions
    let mut y = x.clone();
    for (i, &ax) in lead_axes.iter().chain(axis.iter()).enumerate() {
        y = y.sum_axis(Axis(ax - i)).into_owned();
    }

    // Reshape to the desired output shape and return
    y.to_shape(shape).unwrap().to_owned()
}

/// Performs matrix multiplication between two n-dimensional arrays.
///
/// This function supports matrix multiplication for input arrays with the following dimension combinations:
/// - (1, 1): Scalar-scalar multiplication
/// - (2, 2): Matrix-matrix multiplication
/// - (2, 1): Matrix-vector multiplication (treating the second argument as a column vector)
/// - (1, 2): Vector-matrix multiplication (treating the first argument as a row vector)
///
/// If the input arrays have incompatible dimensions for matrix multiplication, an error message
/// `"Unsupported dimension combination for matmul"` will be returned.
///
/// # Arguments
/// * `a` - The first input array.
///   - Must have type `ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>`.
/// * `b` - The second input array.
///   - Must have type `ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>`.
///
/// # Returns
/// * `Result<Array<f64, Dim<IxDynImpl>>, &'static str>`
///   - On success, returns `Ok` containing the resulting n-dimensional array of type
///     `Array<f64, Dim<IxDynImpl>>`.
///   - On failure due to incompatible dimensions, returns `Err` with the error message
///     `"Unsupported dimension combination for matmul"`.
pub fn matmul(
    a: &ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
    b: &ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
) -> Result<Array<f64, Dim<IxDynImpl>>, &'static str> {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let a = a.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            let b = b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            Ok(array![a.dot(&b)].into_dyn())
        }
        (2, 2) => {
            let a = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            Ok(a.dot(&b).into_dyn())
        }
        (2, 1) => {
            let a = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            let b = b.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            Ok(a.dot(&b).into_dyn())
        }
        (1, 2) => {
            let a = a.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
            let b = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            Ok(a.dot(&b).into_dyn())
        }
        _ => Err("Unsupported dimension combination for matmul"),
    }
}

pub struct WeakKey<T> {
    weak_ref: Weak<RefCell<T>>,
}

impl<T> WeakKey<T> {
    pub fn from(key: &Rc<RefCell<T>>) -> Self {
        WeakKey {
            weak_ref: Rc::downgrade(key),
        }
    }
}

impl<T> PartialEq for WeakKey<T> {
    fn eq(&self, other: &Self) -> bool {
        if let Some(lhs) = &self.weak_ref.upgrade() {
            if let Some(rhs) = &other.weak_ref.upgrade() {
                return Rc::ptr_eq(lhs, rhs);
            }
        }
        false
    }
}

impl<T> Eq for WeakKey<T> {}

impl<T> Hash for WeakKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.weak_ref.upgrade() {
            Some(rc) => Rc::as_ptr(&rc).hash(state),
            None => 0.hash(state),
        }
    }
}

pub fn logsumexp(x: &Array<f64, IxDyn>, axis: usize) -> Array<f64, IxDyn> {
    let axis = Axis(axis);
    let m = x
        .map_axis(axis, |row| {
            row.iter().cloned().fold(f64::MIN, |acc, x| acc.max(x))
        })
        .insert_axis(axis);
    let y = x - &m;
    let y = y.exp();
    let s = y.sum_axis(axis).insert_axis(axis);
    let s = s.ln();
    m + s
}

pub fn one_hot(x: &Variable) -> Variable {
    let x = x.content.borrow();
    let n = x.data.len();
    let v = x
        .data
        .clone()
        .mapv(|x| x.round() as usize)
        .into_raw_vec_and_offset()
        .0;
    let e = ndarray::Array2::eye(n);
    let one_hot = e.select(Axis(0), &v);
    Variable::from_arry(one_hot.into_dyn())
}

mod tests {
    use super::*;

    #[test]
    fn sum_to_test() {
        // Example usage
        let x = Array::from_shape_vec((3, 4, 5), (0..60).map(|v| v as f64).collect()).unwrap();
        let shape = [1, 4, 1];
        let result = sum_to(&x.into_dyn(), &shape);
        println!("Result:\n{:?}", result);
    }

    #[test]
    fn cast_to_fixed_array() {
        let a: ArrayBase<OwnedRepr<f64>, _> =
            ArrayBase::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b: ArrayBase<OwnedRepr<f64>, _> =
            ArrayBase::from_shape_vec(IxDyn(&[3, 2]), vec![7., 8., 9., 10., 11., 12.]).unwrap();

        match matmul(&a, &b) {
            Ok(result) => println!("Result of a.dot(b):\n{:?}", result),
            Err(e) => println!("Error: {}", e),
        }
    }
}
