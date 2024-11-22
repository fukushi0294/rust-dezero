use std::{cell::RefCell, rc::Rc};

use derives::{BiFunction, FunctionNode};
use ndarray::{array, Axis};

use crate::{
    core::{
        function::{self as F, BiFunction, Function, FunctionNode},
        variable::Variable,
    },
    utils,
};

#[derive(BiFunction, FunctionNode)]
pub struct MeanSquaredError {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        MeanSquaredError {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for MeanSquaredError {
    fn forward(
        &self,
        inputs: &[ndarray::Array<f64, ndarray::IxDyn>],
    ) -> Vec<ndarray::Array<f64, ndarray::IxDyn>> {
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        let diff = x0 - x1;
        let n = diff.len() as f64;
        let y = diff.pow2().sum() / n;
        vec![ndarray::array![y].into_dyn()]
    }

    fn backward(
        &self,
        gys: Vec<crate::core::variable::Variable>,
    ) -> Vec<crate::core::variable::Variable> {
        let x0 = self.input0.clone().unwrap();
        let x1 = self.input1.clone().unwrap();
        let diff = x0 - x1;
        let diff_data = diff.data().clone();
        let gy = gys[0].clone();
        let gy = F::bloadcast_to(gy, diff_data.dim());
        let gx0 = gy * diff * (2.0 / diff_data.len() as f64);
        let gx1 = -gx0.clone();
        vec![gx0, gx1]
    }
}

#[derive(BiFunction, FunctionNode)]
pub struct CrossEntropyLoss {
    #[node_I]
    y_pred: Option<Variable>,
    #[node_I]
    y_true: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss {
            y_pred: None,
            y_true: None,
            output: None,
        }
    }
}

impl Function for CrossEntropyLoss {
    fn forward(
        &self,
        inputs: &[ndarray::Array<f64, ndarray::IxDyn>],
    ) -> Vec<ndarray::Array<f64, ndarray::IxDyn>> {
        let x = inputs[0].clone(); // logits
        let n = x.len();
        let y = inputs[1].clone();
        let t_index: Vec<usize> = y.mapv(|x| x.round() as usize).into_raw_vec_and_offset().0; // label index

        let log_z = utils::logsumexp(&x, 1);
        let log_p = x - log_z;
        let tlog_p = log_p.select(Axis(1), &t_index);
        let y = -tlog_p.sum() / (n as f64);
        vec![array![y].into_dyn()]
    }

    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        let gy = gys[0].clone();
        let x = self.y_pred.clone().unwrap().clone(); // logits
        let y = self.y_true.clone().unwrap().clone(); // label
        let y_onehot = utils::one_hot(&y);
        let p = F::Softmax::new(1)(x);
        let gx = gy * (p - y_onehot);
        vec![gx]
    }
}

mod test {

    use ndarray::{Array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use crate::{
        core::{
            function::{self, BiFunction},
            variable::{VarData, Variable},
        },
        loss,
    };

    #[test]
    fn linear_regression() {
        let x = Array::random((100, 1), Uniform::new(0., 1.));
        let y = 5. + 2. * &x + Array::random((100, 1), Uniform::new(0., 1.));
        let (x, y) = (
            VarData::new(x.into_dyn()).to_node(),
            VarData::new(y.into_dyn()).to_node(),
        );

        let mut w = VarData::new(Array2::zeros((1, 1)).into_dyn()).to_node();
        let mut b = VarData::new(Array1::zeros(1).into_dyn()).to_node();

        fn predict(x: Variable, w: Variable, b: Variable) -> Variable {
            function::matmal(x, w) + b
        }

        let lr = 0.1;

        for _ in 0..100 {
            let y_pred = predict(x.clone(), w.clone(), b.clone());
            let mut loss = loss::MeanSquaredError::new().apply(y.clone(), y_pred);
            w.cleargrad();
            b.cleargrad();
            loss.backward();
            let w_new = w.data() - lr * w.grad().unwrap().data();
            w.set_data(w_new);
            let b_new = b.data() - lr * b.grad().unwrap().data();
            b.set_data(b_new);
            println!("{}, {}, {}", w, b, loss)
        }
    }
}
