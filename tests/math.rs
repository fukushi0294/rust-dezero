mod tests {
    use rust_dezero::core::variable::Variable;

    #[test]
    fn sphere_function() {
        let x = Variable::from_vec1(vec![1.0]).to_node();
        let y = Variable::from_vec1(vec![1.0]).to_node();
        let z = x.powi(2) + y.powi(2);
        z.backward();
        let x_grad = x.get_grad_vec();
        let y_grad = y.get_grad_vec();
        assert_eq!(vec![2.0], x_grad);
        assert_eq!(vec![2.0], y_grad);
    }

    #[test]
    fn matyas_function() {
        let x = Variable::from_vec1(vec![1.0]).to_node();
        let y = Variable::from_vec1(vec![1.0]).to_node();
        let z = 0.26 * (x.clone().powi(2) + y.clone().powi(2)) - 0.48 * x.clone() * y.clone();
        println!("{}", z);
        z.backward();
        let x_grad = x.grad().unwrap();
        println!("{}", x_grad);
    }

    #[test]
    fn second_derivative() {
        let x = Variable::from_vec1(vec![2.0]).to_node();
        let y = x.clone().powi(4) - 2.0 * x.clone().powi(2);
        y.enable_graph();
        y.backward();
        assert_eq!(vec![24.0], x.get_grad_vec());
        let grad_node = x.grad().unwrap();
        x.cleargrad();
        grad_node.backward();
        assert_eq!(vec![44.0], x.get_grad_vec())
    }

    #[test]
    fn newton() {
        let x = Variable::from_vec1(vec![2.0]).to_node();
        for _ in 1..10 {
            println!("{}", x);
            let y = x.clone().powi(4) - 2.0 * x.clone().powi(2);
            x.cleargrad();
            y.enable_graph();
            y.backward();
            let gx = x.grad().unwrap();
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
}
