use crate::core::config::CONFIG;
use crate::core::variable::{VarNode, Variable};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::i32;
use std::rc::Rc;

pub trait Function {
    fn call(&mut self, inputs: &[Rc<RefCell<Variable>>]) -> Vec<Rc<RefCell<Variable>>> {
        let mut x = Vec::new();
        for v in inputs.iter() {
            let var = v.borrow().clone();
            x.push(var.data.clone());
        }
        let ys = self.forward(x.as_slice());
        let mut outputs = Vec::new();
        let mut refs = Vec::new();
        for y in ys {
            let output_ref = Rc::new(RefCell::new(Variable::new(y)));
            refs.push(output_ref.clone());
            outputs.push(output_ref);
        }

        if CONFIG.lock().unwrap().enable_backprop {
            let function_node = self.new_instance(inputs, &refs);
            for output in outputs.iter_mut() {
                output.borrow_mut().creator = Some(function_node.clone());
            }
        }
        outputs
    }
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function>;
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>>;
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode>;
    fn supplyer(&self) -> Rc<dyn ParamSupplier>;
}

pub trait ParamSupplier {
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>>;
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>>;
}

struct UniFunctionParamSupplier {
    input: Rc<RefCell<Variable>>,
    output: Rc<RefCell<Variable>>,
}

pub trait UniFunction: Function {
    fn apply(&mut self, input: VarNode) -> VarNode {
        let output = self.call(&[input.content]);
        VarNode {
            content: output.get(0).unwrap().clone(),
        }
    }
}

impl ParamSupplier for UniFunctionParamSupplier {
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        vec![Rc::clone(&self.input.clone())]
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        vec![Rc::clone(&self.output.clone())]
    }
}

pub trait BiFunction: Function {
    fn apply(&mut self, input0: VarNode, input1: VarNode) -> VarNode {
        let output = self.call(&[input0.content, input1.content]);
        VarNode {
            content: output.get(0).unwrap().clone(),
        }
    }
}

struct BiFunctionParamSupplier {
    input: (Rc<RefCell<Variable>>, Rc<RefCell<Variable>>),
    output: Rc<RefCell<Variable>>,
}

impl ParamSupplier for BiFunctionParamSupplier {
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        let x1 = self.input.0.clone();
        let x2 = self.input.1.clone();
        vec![x1, x2]
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        vec![self.output.clone()]
    }
}

macro_rules! params {
    (($input:expr), ($output:expr)) => {
        std::rc::Rc::new($crate::core::function::UniFunctionParamSupplier {
            input: $input.clone(),
            output: $output.clone(),
        })
    };

    (($input1:expr, $input2:expr), ($output:expr)) => {
        std::rc::Rc::new($crate::core::function::BiFunctionParamSupplier {
            input: ($input1.clone(), $input2.clone()),
            output: $output.clone(),
        })
    };
}

pub struct Square {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Square {
    pub fn new() -> Self {
        Square {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Square {}

impl Function for Square {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Square {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.pow2()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = VarNode { content: v.clone() };
            return vec![2.0 * x * gys.get(0).unwrap().clone()];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub fn square(x: Rc<RefCell<Variable>>) -> Vec<Rc<RefCell<Variable>>> {
    let mut f = Square::new();
    f.call(&[x])
}

pub struct Exp {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Exp {
    pub fn new() -> Self {
        Exp {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Exp {}

impl Function for Exp {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Exp {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.exp()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = VarNode { content: v.clone() };
            return vec![x.exp() * gys.get(0).unwrap().clone()];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub fn exp(x: Rc<RefCell<Variable>>) -> Vec<Rc<RefCell<Variable>>> {
    let mut f = Exp::new();
    f.call(&[x])
}

pub struct Add {
    input: (Option<Rc<RefCell<Variable>>>, Option<Rc<RefCell<Variable>>>),
    output: Option<Rc<RefCell<Variable>>>,
}

impl Add {
    pub fn new() -> Self {
        Add {
            input: (None, None),
            output: None,
        }
    }
}

impl BiFunction for Add {}

impl Function for Add {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x1 = inputs.get(0).unwrap().clone();
        let x2 = inputs.get(1).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Add {
            input: (Some(x1), Some(x2)),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x1 = inputs[0].clone();
        let x2 = inputs[1].clone();
        vec![x1 + x2]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        return vec![gys[0].clone(), gys[0].clone()];
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.0.clone().unwrap(), self.input.1.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Mul {
    input: (Option<Rc<RefCell<Variable>>>, Option<Rc<RefCell<Variable>>>),
    output: Option<Rc<RefCell<Variable>>>,
}

impl Mul {
    pub fn new() -> Self {
        Mul {
            input: (None, None),
            output: None,
        }
    }
}

impl BiFunction for Mul {}

impl Function for Mul {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x1 = inputs.get(0).unwrap().clone();
        let x2 = inputs.get(1).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Mul {
            input: (Some(x1), Some(x2)),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x1 = inputs[0].clone();
        let x2 = inputs[1].clone();
        vec![x1 * x2]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = VarNode {
                content: self.input.0.clone().unwrap().clone(),
            };
            let x2 = VarNode {
                content: self.input.1.clone().unwrap().clone(),
            };
            vec![gys[0].clone() * x2, gys[0].clone() * x1]
        } else {
            vec![]
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.0.clone().unwrap(), self.input.1.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Sub {
    input: (Option<Rc<RefCell<Variable>>>, Option<Rc<RefCell<Variable>>>),
    output: Option<Rc<RefCell<Variable>>>,
}

impl Sub {
    pub fn new() -> Self {
        Sub {
            input: (None, None),
            output: None,
        }
    }
}

impl BiFunction for Sub {}

impl Function for Sub {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x1 = inputs.get(0).unwrap().clone();
        let x2 = inputs.get(1).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Sub {
            input: (Some(x1), Some(x2)),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x1 = inputs[0].clone();
        let x2 = inputs[1].clone();
        vec![x1 - x2]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        return vec![gys[0].clone(), -gys[0].clone()];
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.0.clone().unwrap(), self.input.1.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Div {
    input: (Option<Rc<RefCell<Variable>>>, Option<Rc<RefCell<Variable>>>),
    output: Option<Rc<RefCell<Variable>>>,
}

impl Div {
    pub fn new() -> Self {
        Div {
            input: (None, None),
            output: None,
        }
    }
}

impl BiFunction for Div {}

impl Function for Div {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x1 = inputs.get(0).unwrap().clone();
        let x2 = inputs.get(1).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Mul {
            input: (Some(x1), Some(x2)),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x1 = inputs[0].clone();
        let x2 = inputs[1].clone();
        vec![x1 / x2]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = VarNode {
                content: self.input.0.clone().unwrap().clone(),
            };
            let x2 = VarNode {
                content: self.input.1.clone().unwrap().clone(),
            };
            vec![
                gys[0].clone() / x1.clone(),
                -gys[0].clone() * x1 / x2.powi(2),
            ];
        }
        vec![]
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.0.clone().unwrap(), self.input.1.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Neg {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Neg {
    pub fn new() -> Self {
        Neg {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Neg {}

impl Function for Neg {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Neg {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![-x]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.input {
            return vec![-gys[0].clone()];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Pow {
    input: Option<Rc<RefCell<Variable>>>,
    factor: i32,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Pow {
    pub fn new(factor: i32) -> Self {
        Pow {
            input: None,
            factor,
            output: None,
        }
    }
}

impl UniFunction for Pow {}

impl Function for Pow {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Pow {
            input: Some(x),
            factor: self.factor,
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.powi(self.factor)]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = VarNode { content: v.clone() };
            return vec![self.factor as f64 * x.powi(self.factor - 1) * gys[0].clone()];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Sin {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Sin {
    pub fn new() -> Self {
        Sin {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Sin {}

impl Function for Sin {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Sin {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.sin()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = VarNode { content: v.clone() };
            return vec![gys[0].clone() * Cos::new().apply(x)];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Cos {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Cos {
    pub fn new() -> Self {
        Cos {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Cos {}

impl Function for Cos {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Cos {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.cos()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = VarNode { content: v.clone() };
            return vec![-Sin::new().apply(x) * gys[0].clone()];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Tanh {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Tanh {}

impl Function for Tanh {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Tanh {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![(x.exp() - x.exp().powi(-1)) / (x.exp() + x.exp().powi(-1))]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.output {
            let y = VarNode { content: v.clone() };
            let gy = gys.get(0).unwrap().clone();
            return vec![gy * (1.0 - y.powi(2))];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

pub struct Reshape {
    shape: Vec<usize>,
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Reshape {
            shape: shape,
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Reshape {}

impl Function for Reshape {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Reshape {
            shape: self.shape.clone(),
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        let reshaped = x.to_shape(self.shape.clone());
        vec![reshaped.unwrap().to_owned()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.output {
            let gy = gys.get(0).unwrap();
            let shape = self
                .input
                .clone()
                .unwrap()
                .borrow()
                .data
                .shape()
                .to_vec();
            return vec![reshape(gy.clone(), shape)];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

fn reshape(x: VarNode, shape: Vec<usize>) -> VarNode {
    if x.data().shape() == shape {
        x
    } else {
        Reshape::new(shape).apply(x)
    }
}

pub struct Transpose {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Transpose {
    pub fn new() -> Self {
        Transpose {
            input: None,
            output: None,
        }
    }
}

impl UniFunction for Transpose {}

impl Function for Transpose {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs.get(0).unwrap().clone();
        let y = outputs.get(0).unwrap().clone();
        let f = Transpose {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.t().to_owned()]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.output {
            let gy = gys.get(0).unwrap().clone();
            return vec![transpose(gy)];
        } else {
            return vec![];
        }
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}

fn transpose(x: VarNode) -> VarNode {
    Transpose::new().apply(x)
}

fn numerical_diff(f: &mut impl Function, x: Variable) -> Array<f64, IxDyn> {
    let eps = 1e-4;
    let x0 = Variable::new(x.data.clone() - eps);
    let x1 = Variable::new(x.data.clone() + eps);
    let y0 = f.call(&[Rc::new(RefCell::new(x0))]);
    let y0_data = y0.get(0).unwrap().borrow_mut().data.clone();
    let y1 = f.call(&[Rc::new(RefCell::new(x1))]);
    let y1_data = y1.get(0).unwrap().borrow_mut().data.clone();
    return (y1_data - y0_data) / (eps * 2.0);
}

#[cfg(test)]
mod tests {
    use crate::no_grad;
    use ndarray::Array1;

    use super::*;

    #[test]
    fn square_test() {
        let x1 = vec![5.0, 10.0];
        let expected: Vec<f64> = x1.iter().map(|&x| x * x).collect();
        let x2 = Variable::new(Array1::from_vec(x1).into_dyn());
        let mut f = Square::new();
        let actual = f.call(&[Rc::new(RefCell::new(x2))]);
        assert_eq!(1, actual.len());
        let mut result = Vec::new();
        let var = actual.get(0).unwrap().borrow_mut();
        for data in var.data.clone() {
            result.push(data);
        }
        assert_eq!(expected, result);
    }

    #[test]
    fn same_var_test() {
        let x = Variable::from_vec1(vec![3.0]).to_node();
        let mut add = Add::new();
        let y = add.apply(x.clone(), x.clone());
        y.backward();
        let grad = x.get_grad_vec();
        assert_eq!(vec![2.0], grad)
    }

    #[test]
    fn backward_test() {
        let input = vec![10.0];
        let x = Variable::from_vec1(input.clone()).to_node();
        let expected: Vec<f64> = input.iter().map(|x| 2.0 * x).collect();
        let mut f = Square::new();
        let y = f.apply(x.clone());
        y.backward();
        let grad = x.get_grad_vec();
        assert_eq!(expected, grad)
    }

    #[test]
    fn calc_graph_test() {
        let x = Variable::from_vec1(vec![2.0]).to_node();
        let mut square = Square::new();
        let mut add = Add::new();
        let a = square.apply(x.clone());
        let (a1, a2) = (a.clone(), a.clone());
        let a1 = square.apply(a1);
        let a2 = square.apply(a2);
        let y = add.apply(a1, a2);
        y.backward();
        println!("{}", y.content.borrow().data);
        let grad = x.get_grad_vec();
        assert_eq!(vec![64.0], grad)
    }

    #[test]
    fn forward_only_test() {
        no_grad! {

            let x1 = vec![5.0, 10.0];
            let expected: Vec<f64> = x1.iter().map(|&x| x * x).collect();
            let x2 = Variable::new(Array1::from_vec(x1).into_dyn());
            let mut f = Square::new();
            let actual = f.call(&[Rc::new(RefCell::new(x2))]);
            assert_eq!(1, actual.len());
            let mut result = Vec::new();
            let var = actual.get(0).unwrap().borrow_mut();
            for data in var.data.clone() {
                result.push(data);
            }
            assert_eq!(expected, result);
        }
    }
}
