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
        let (x1, x2) = x.clone().blanch();
        let (y1, y2) = y.clone().blanch();
        let out = 0.26 * (x1.powi(2) + y1.powi(2)) - 0.48 * x2 * y2; 
        let mut z = out.extract();
        println!("{}", z.data);
        z.backward();
        let x_grad = x.get_grad_vec();
        println!("{:?}", x_grad);
    }
}