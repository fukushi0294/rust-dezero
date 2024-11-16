mod ml {
    use std::{collections::HashSet, f64::consts::PI};

    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use rust_dezero::{
        core::{
            function::{sigmoid, BiFunction},
            variable::{VarNode, Variable},
        },
        loss,
        nn::{self, Layer},
        optimizer::{Optimizer, SGD},
    };

    struct TwoLayerNet {
        l1: nn::Linear,
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
        fn forward(&self, x: VarNode) -> VarNode {
            let y = self.l1.forward(x);
            let y = sigmoid(y);
            let y = self.l2.forward(y);
            y
        }

        fn parameters(&self) -> HashSet<VarNode> {
            let mut set = HashSet::new();
            for node in self
                .l1
                .parameters()
                .iter()
                .chain(self.l2.parameters().iter())
            {
                set.insert(node.clone());
            }
            set
        }
    }

    #[test]
    fn two_layer_model() {
        let x = Array::random((100, 1), Uniform::new(0., 1.));
        let y = (2. * PI * x.clone()).sin() + Array::random((100, 1), Uniform::new(0., 1.));

        let x = Variable::new(x.into_dyn()).to_node();
        let y = Variable::new(y.into_dyn()).to_node();

        let input_size = 100;
        let hidden_size = 10;
        let output_size = 1;

        let model = TwoLayerNet::new(input_size, hidden_size, output_size);

        let max_iter = 10000;
        let lr = 0.2;

        let optimizer = SGD::new(lr, model.parameters());
        for i in 0..max_iter {
            let y_pred = model.forward(x.clone());
            let loss = loss::MeanSquaredError::new().apply(y.clone(), y_pred);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if i % 1000 == 0 {
                println!("{}", loss)
            }
        }
    }
}
