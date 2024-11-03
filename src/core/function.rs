use crate::core::config::CONFIG;
use crate::core::variable::{PlaceHolder, Variable};
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::i32;
use std::rc::Rc;

pub trait Function {
    fn call(&mut self, inputs: &[Rc<RefCell<Variable>>]) -> Vec<Rc<RefCell<Variable>>> {
        let mut x = Vec::new();
        for v in inputs.iter() {
            let var = v.borrow_mut().clone();
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

    fn apply(&mut self, inputs: PlaceHolder) -> PlaceHolder {
        PlaceHolder {
            content: self.call(&inputs.content),
        }
    }
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function>;
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>>;
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>>;
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>>;
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>>;
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.borrow_mut().data.clone();
            return vec![2.0 * x * &gys[0]];
        } else {
            return vec![];
        }
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.input {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.borrow_mut().data.clone();
            return vec![x.exp() * &gys[0]];
        } else {
            return vec![];
        }
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.input {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        return vec![gys[0].clone(), gys[0].clone()];
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = &self.input.0.clone().unwrap();
            let x2 = &self.input.1.clone().unwrap();
            vec![Rc::clone(x1), Rc::clone(x2)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = self.input.0.clone().unwrap().borrow_mut().data.clone();
            let x2 = self.input.1.clone().unwrap().borrow_mut().data.clone();
            vec![gys[0].clone() * x1, gys[0].clone() * x2];
        }
        vec![]
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = &self.input.0.clone().unwrap();
            let x2 = &self.input.1.clone().unwrap();
            vec![Rc::clone(x1), Rc::clone(x2)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
        vec![x1 + x2]
    }
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        return vec![gys[0].clone(), -gys[0].clone()];
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = &self.input.0.clone().unwrap();
            let x2 = &self.input.1.clone().unwrap();
            vec![Rc::clone(x1), Rc::clone(x2)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = self.input.0.clone().unwrap().borrow_mut().data.clone();
            let x2 = self.input.1.clone().unwrap().borrow_mut().data.clone();
            vec![gys[0].clone() / &x1, -gys[0].clone() * &x1 / x2.pow2()];
        }
        vec![]
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if self.input.0.is_some() && self.input.1.is_some() {
            let x1 = &self.input.0.clone().unwrap();
            let x2 = &self.input.1.clone().unwrap();
            vec![Rc::clone(x1), Rc::clone(x2)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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

impl Function for Neg {
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
        vec![-x]
    }
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            return vec![-gys[0].clone()];
        } else {
            return vec![];
        }
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.input {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
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
    fn backward(&self, gys: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.borrow_mut().data.clone();
            return vec![self.factor as f64 * x.powi(self.factor -1) * gys[0].clone()];
        } else {
            return vec![];
        }
    }
    fn get_inputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.input {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
    }
    fn get_outputs(&self) -> Vec<Rc<RefCell<Variable>>> {
        if let Some(v) = &self.output {
            vec![Rc::clone(v)]
        } else {
            vec![]
        }
    }
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
    fn backward_test() {
        let x = Array1::from_vec(vec![10.0]);
        let expected: Vec<f64> = x.to_vec().iter().map(|x| 2.0 * x).collect();
        let mut f = Square::new();
        let input = Rc::new(RefCell::new(Variable::new(x.into_dyn())));
        let y = f.call(&[input]);
        let mut var = y.get(0).unwrap().borrow_mut();
        let result = var.backward();
        assert_eq!(true, result.is_some());
        let binding = result.unwrap();
        let input_var = binding.borrow_mut();
        let grad = input_var.grad.clone().unwrap().into_raw_vec_and_offset().0;
        assert_eq!(expected, grad)
    }

    #[test]
    fn calc_graph_test() {
        let x = Variable::new(Array1::from_vec(vec![2.0]).into_dyn());
        let x = Rc::new(RefCell::new(x));
        let mut square = Square::new();
        let mut add = Add::new();
        let a = square.call(&[x]);
        let a1 = square.call(&a);
        let a2 = square.call(&a);
        let a1_ = a1.get(0).unwrap().clone();
        let a2_ = a2.get(0).unwrap().clone();
        let binding = add.call(&[a1_, a2_]);
        let mut y = binding.get(0).unwrap().borrow_mut();
        assert_eq!(vec![32.0], y.data.clone().into_raw_vec_and_offset().0);
        let result = y.backward();
        let binding = result.unwrap();
        let input_var = binding.borrow_mut();
        let grad = input_var.grad.clone().unwrap().into_raw_vec_and_offset().0;
        assert_eq!(vec![64.0], grad)
    }

    #[test]
    fn forward_only_test() {
        no_grad!();
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
