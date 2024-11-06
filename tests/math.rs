mod tests {
    use rust_dezero::core::variable::Variable;

    #[test]
    fn sphere_function() {
        let x = Variable::from_vec1(vec![1.0]).to_node();
        let y = Variable::from_vec1(vec![1.0]).to_node();
        let z = x.powi(2) + y.powi(2);
        z.extract().backward();
        let x_grad = x.get_grad_vec();
        let y_grad = y.get_grad_vec();
        assert_eq!(vec![2.0], x_grad);
        assert_eq!(vec![2.0], y_grad);
    }

    #[test]
    fn matyas_function() {
        let x = Variable::from_vec1(vec![1.0]).to_node();
        let y = Variable::from_vec1(vec![1.0]).to_node();
        let out = 0.26 * (x.clone().powi(2) + y.clone().powi(2)) - 0.48 * x.clone() * y.clone(); 
        let mut z = out.extract();
        println!("{}", z.data);
        z.backward();
        let x_grad = x.get_grad_vec();
        println!("{:?}", x_grad);
    }

    #[test]
    fn second_derivative() {
        let x = Variable::from_vec1(vec![2.0]).to_node();
        let y = x.clone().powi(4) - 2.0 * x.clone().powi(2);
        let mut  y = y.extract();
        y.create_graph = true;
        y.backward();
        assert_eq!(vec![24.0], x.get_grad_vec());
        let grad_node = x.extract().grad.clone().unwrap();
        x.extract().cleargrad();
        grad_node.content.borrow_mut().backward();
        assert_eq!(vec![44.0], x.get_grad_vec())
    }
}