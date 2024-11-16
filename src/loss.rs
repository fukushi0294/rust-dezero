use std::{cell::RefCell, rc::Rc};

use derives::BiFunction;

use crate::{
    core::{
        function::{self, BiFunction, Function, ParamSupplier},
        variable::{VarNode, Variable},
    },
    params,
};

#[derive(BiFunction)]
pub struct MeanSquaredError {
    input: (Option<Rc<RefCell<Variable>>>, Option<Rc<RefCell<Variable>>>),
    output: Option<Rc<RefCell<Variable>>>,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        MeanSquaredError {
            input: (None, None),
            output: None,
        }
    }
}

impl Function for MeanSquaredError {
    fn new_instance(
        &self,
        inputs: &[std::rc::Rc<std::cell::RefCell<crate::core::variable::Variable>>],
        outputs: &[std::rc::Rc<std::cell::RefCell<crate::core::variable::Variable>>],
    ) -> std::rc::Rc<dyn Function> {
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        let y = outputs[0].clone();
        let f = MeanSquaredError {
            input: (Some(x0), Some(x1)),
            output: Some(y),
        };
        Rc::new(f)
    }

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
        gys: Vec<crate::core::variable::VarNode>,
    ) -> Vec<crate::core::variable::VarNode> {
        let x0 = VarNode {
            content: self.input.0.clone().unwrap(),
        };
        let x1 = VarNode {
            content: self.input.1.clone().unwrap(),
        };
        let diff = x0 - x1;
        let diff_data = diff.data().clone();
        let gy = gys[0].clone();
        let gy = function::bloadcast_to(gy, diff_data.dim());
        let gx0 = gy * diff * (2.0 / diff_data.len() as f64);
        let gx1 = -gx0.clone();
        vec![gx0, gx1]
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.0.clone().unwrap(), self.input.1.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

mod test {

    use ndarray::{Array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use crate::{
        core::{
            function::{self, BiFunction},
            variable::{VarNode, Variable},
        },
        loss,
    };

    #[test]
    fn linear_regression() {
        let x = Array::random((100, 1), Uniform::new(0., 1.));
        let y = 5. + 2. * &x + Array::random((100, 1), Uniform::new(0., 1.));
        let (x, y) = (
            Variable::new(x.into_dyn()).to_node(),
            Variable::new(y.into_dyn()).to_node(),
        );

        let w = Variable::new(Array2::zeros((1, 1)).into_dyn()).to_node();
        let b = Variable::new(Array1::zeros(1).into_dyn()).to_node();

        fn predict(x: VarNode, w: VarNode, b: VarNode) -> VarNode {
            function::matmal(x, w) + b
        }

        let lr = 0.1;

        for _ in 0..100 {
            let y_pred = predict(x.clone(), w.clone(), b.clone());
            let loss = loss::MeanSquaredError::new().apply(y.clone(), y_pred);
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
