use ndarray::{Array, Axis, IxDyn};

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
}
