mod ml {
    use std::f64::consts::PI;

    use derives::Learnable;
    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rust_dezero::{
        core::{
            function::{sigmoid, BiFunction},
            variable::{Variable},
        },
        loss,
        nn::{self, Layer, Learnable},
        optimizer::{MomentumSGD, Optimizer, SGD},
    };

    #[test]
    fn linear_layer_test() {
        let x = Array::random((100, 1), Uniform::new(0., 1.));
        let y = 5. + 2. * &x + Array::random((100, 1), Uniform::new(0., 1.));
        let (x, y) = (
            Variable::from_arry(x.into_dyn()),
            Variable::from_arry(y.into_dyn()),
        );
        let model = nn::Linear::new(1, 1);
        let lr = 0.1;
        let mut optimizer = SGD::new(lr, model.parameters());
        for _ in 0..100 {
            let y_pred = model.forward(x.clone());
            let mut loss = loss::MeanSquaredError::new().apply(y.clone(), y_pred);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            println!("{}", loss)
        }
    }

    #[derive(Learnable)]
    struct TwoLayerNet {
        #[learnable]
        l1: nn::Linear,
        #[learnable]
        l2: nn::Linear,
    }

    impl TwoLayerNet {
        pub fn new(in_size: usize, hidden: usize, out_size: usize) -> Self {
            TwoLayerNet {
                l1: nn::Linear::new(in_size, hidden),
                l2: nn::Linear::new(hidden, out_size),
            }
        }
    }

    impl nn::Layer for TwoLayerNet {
        fn forward(&self, x: Variable) -> Variable {
            let y = self.l1.forward(x);
            let y = sigmoid(y);
            let y = self.l2.forward(y);
            y
        }
    }

    #[test]
    fn two_layer_model() {
        let x = Array::random((100, 1), Uniform::new(0., 1.));
        let y = (2. * PI * x.clone()).sin() + Array::random((100, 1), Uniform::new(0., 1.));

        let x = Variable::from_arry(x.into_dyn());
        let y = Variable::from_arry(y.into_dyn());

        let input_size = 1;
        let hidden_size = 10;
        let output_size = 1;

        let model = TwoLayerNet::new(input_size, hidden_size, output_size);

        let max_iter = 1000;
        let lr = 0.01;
        let mut criterion = loss::MeanSquaredError::new();

        let mut optimizer = MomentumSGD::new(lr, 0.9, model.parameters());
        for i in 0..max_iter {
            let y_pred = model.forward(x.clone());
            let mut loss = criterion(y.clone(), y_pred);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            if i % 100 == 0 {
                println!("{}", loss)
            }
        }
    }
}
