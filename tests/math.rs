mod tests {
    use ndarray::array;
    use rust_dezero::core::{
        function::{Cos, Sin, UniFunction},
        variable::{VarData, Variable},
    };

    fn assert_almost_equal(a: f64, b: f64) {
        let epsilon = 1e-8;
        assert!(
            (a - b).abs() < epsilon,
            "Values are not approximately equal: {} vs {}",
            a,
            b
        );
    }

    #[test]
    fn sphere_function() {
        let x = Variable::from_arry(array![1.0].into_dyn());
        let y = Variable::from_arry(array![1.0].into_dyn());
        let mut z = x.powi(2) + y.powi(2);
        z.backward();
        let x_grad = x.get_grad_vec();
        let y_grad = y.get_grad_vec();
        assert_eq!(vec![2.0], x_grad);
        assert_eq!(vec![2.0], y_grad);
    }

    #[test]
    fn matyas_function() {
        let x = Variable::from_arry(array![1.0].into_dyn());
        let y = Variable::from_arry(array![1.0].into_dyn());
        let mut z = 0.26 * (x.clone().powi(2) + y.clone().powi(2)) - 0.48 * x.clone() * y.clone();
        println!("{}", z);
        z.backward();
        let x_grad = x.grad().unwrap();
        println!("{}", x_grad);
    }

    #[test]
    fn second_derivative() {
        let mut x = Variable::from_arry(array![2.0].into_dyn());
        let mut y = x.clone().powi(4) - 2.0 * x.clone().powi(2);
        y.enable_graph();
        y.backward();
        assert_eq!(vec![24.0], x.get_grad_vec());
        let mut grad_node = x.grad().unwrap();
        x.cleargrad();
        grad_node.backward();
        assert_eq!(vec![44.0], x.get_grad_vec())
    }

    #[test]
    fn newton() {
        let mut x = Variable::from_arry(array![2.0].into_dyn());
        for _ in 1..10 {
            println!("{}", x);
            let mut y = x.clone().powi(4) - 2.0 * x.clone().powi(2);
            x.cleargrad();
            y.enable_graph();
            y.backward();
            let mut gx = x.grad().unwrap();
            let gx1 = x.grad().unwrap().data();
            x.cleargrad();
            gx.backward();
            let gx2 = x.grad().unwrap().data();
            let delta = gx1 / gx2;
            let data = x.data().clone();
            x.set_data(data - delta);
        }
        let x_min = x.data().into_raw_vec_and_offset().0;
        assert_eq!(vec![1.0], x_min)
    }

    #[test]
    fn sin_functions() {
        let x = Variable::from_arry(array![std::f64::consts::PI].into_dyn());
        let mut y = Sin::new()(x.clone());
        let binding = y.data().into_raw_vec_and_offset();
        let result = binding.0.get(0).unwrap();
        assert_almost_equal(0., *result);
        y.backward();
        assert_eq!(vec![-1.0], x.get_grad_vec())
    }

    #[test]
    fn cos_functions() {
        let x = Variable::from_arry(array![std::f64::consts::PI / 2.0].into_dyn());
        let mut y = Cos::new().apply(x.clone());
        let binding = y.data().into_raw_vec_and_offset();
        let result = binding.0.get(0).unwrap();
        assert_almost_equal(0., *result);
        y.backward();
        assert_eq!(vec![-1.0], x.get_grad_vec())
    }

    #[test]
    fn tensor_test() {
        let mut a = Variable::from_arry(array![[1., 2., 3.], [4., 5., 6.]].into_dyn());
        let mut b = a.transpose();
        let mut c = a.reshape(vec![3, 2]);
        println!("{}", b);
        println!("{}", c);
        b.backward();
        println!("{}", a.grad().unwrap());
        a.cleargrad();
        c.backward();
        println!("{}", a.grad().unwrap());
    }
}
