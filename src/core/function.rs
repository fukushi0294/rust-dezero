use crate::core::config::CONFIG;
use crate::core::variable::{VarData, Variable};
use crate::nn::Sigmoid;
use crate::utils;
use derives::{BiFunction, FunctionNode, UniFunction};
use ndarray::{Array, Dim, Dimension, IxDyn, IxDynImpl};
use std::rc::Rc;
use std::{i32, usize};

pub trait FunctionNode {
    fn new_instance(&self, inputs: &[Variable], outputs: &[Variable]) -> Rc<dyn Function>;
    fn get_inputs(&self) -> Vec<Variable>;
    fn get_outputs(&self) -> Vec<Variable>;
}

pub trait Function: FunctionNode {
    fn call(&mut self, inputs: &[Variable]) -> Vec<Variable> {
        let mut x = Vec::new();
        for v in inputs.iter() {
            x.push(v.data().clone());
        }
        let ys = self.forward(x.as_slice());
        let mut outputs = Vec::new();
        let mut refs = Vec::new();
        for y in ys {
            let output_ref = Variable::from(VarData::new(y));
            refs.push(output_ref.clone());
            outputs.push(output_ref);
        }

        if CONFIG.lock().unwrap().enable_backprop {
            let function_node = self.new_instance(inputs, &refs);
            for output in outputs.iter_mut() {
                output.creator = Some(function_node.clone());
            }
        }
        outputs
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>>;
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable>;
}

pub trait UniFunction: Function {
    fn apply(&mut self, input: Variable) -> Variable {
        let output = self.call(&[input]);
        output[0].clone()
    }
}

pub trait BiFunction: Function {
    fn apply(&mut self, input0: Variable, input1: Variable) -> Variable {
        let output = self.call(&[input0, input1]);
        output[0].clone()
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Square {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
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
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.pow2()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(x) = &self.input {
            return vec![2.0 * x.clone() * gys[0].clone()];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Exp {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
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
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.exp()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.clone();
            return vec![x.exp() * gys[0].clone()];
        } else {
            return vec![];
        }
    }
}

pub fn exp(x: Variable) -> Variable {
    let mut f = Exp::new();
    f(x)
}

#[derive(BiFunction, FunctionNode)]
pub struct Add {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Add {
    pub fn new() -> Self {
        Add {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for Add {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        vec![x0 + x1]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        let (mut gx0, mut gx1) = (gys[0].clone(), gys[0].clone());
        let x0 = self.input0.clone().unwrap();
        let binding = x0.data();
        let x0_shape = binding.shape();
        let x1 = self.input1.clone().unwrap();
        let binding = x1.data();
        let x1_shape = binding.shape();
        if x0_shape != x1_shape {
            gx0 = sum_to(gx0, IxDyn(x0_shape));
            gx1 = sum_to(gx1, IxDyn(x1_shape));
        }
        return vec![gx0, gx1];
    }
}

#[derive(BiFunction, FunctionNode)]
pub struct Mul {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Mul {
    pub fn new() -> Self {
        Mul {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for Mul {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        vec![x0 * x1]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input0.is_some() && self.input1.is_some() {
            let x0 = self.input0.clone().unwrap().clone();
            let x1 = self.input1.clone().unwrap().clone();
            vec![gys[0].clone() * x1, gys[0].clone() * x0]
        } else {
            vec![]
        }
    }
}

#[derive(BiFunction, FunctionNode)]
pub struct MatMul {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl MatMul {
    pub fn new() -> Self {
        MatMul {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for MatMul {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        let y = utils::matmul(&x0, &x1).unwrap();
        vec![y]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        let gy = gys[0].clone();
        let x = self.input0.clone().unwrap().clone();
        let w = self.input1.clone().unwrap().clone();
        let gx = matmal(gy.clone(), w.transpose());
        let gw = matmal(x.transpose(), gy);
        vec![gx, gw]
    }
}

pub fn matmal(x: Variable, w: Variable) -> Variable {
    MatMul::new().apply(x, w)
}

#[derive(BiFunction, FunctionNode)]
pub struct Sub {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Sub {
    pub fn new() -> Self {
        Sub {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for Sub {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x1 = inputs[0].clone();
        let x2 = inputs[1].clone();
        vec![x1 - x2]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        return vec![gys[0].clone(), -gys[0].clone()];
    }
}

#[derive(BiFunction, FunctionNode)]
pub struct Div {
    #[node_I]
    input0: Option<Variable>,
    #[node_I]
    input1: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Div {
    pub fn new() -> Self {
        Div {
            input0: None,
            input1: None,
            output: None,
        }
    }
}

impl Function for Div {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 2, "inputs slice size must be 2");
        let x0 = inputs[0].clone();
        let x1 = inputs[1].clone();
        vec![x0 / x1]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if self.input0.is_some() && self.input1.is_some() {
            let x0 = self.input0.clone().unwrap().clone();
            let x1 = self.input1.clone().unwrap().clone();
            vec![
                gys[0].clone() / x0.clone(),
                -gys[0].clone() * x0 / x1.powi(2),
            ];
        }
        vec![]
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Neg {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
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
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![-x]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.input {
            return vec![-gys[0].clone()];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Pow {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
    factor: i32,
}

impl Pow {
    pub fn new(factor: i32) -> Self {
        Pow {
            input: None,
            output: None,
            factor,
        }
    }
}

impl Function for Pow {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.powi(self.factor)]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.clone();
            return vec![self.factor as f64 * x.powi(self.factor - 1) * gys[0].clone()];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Sin {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Sin {
    pub fn new() -> Self {
        Sin {
            input: None,
            output: None,
        }
    }
}

impl Function for Sin {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.sin()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.clone();
            return vec![gys[0].clone() * Cos::new()(x)];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Cos {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Cos {
    pub fn new() -> Self {
        Cos {
            input: None,
            output: None,
        }
    }
}

impl Function for Cos {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.cos()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.input {
            let x = v.clone();
            return vec![-Sin::new()(x) * gys[0].clone()];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Tanh {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh {
            input: None,
            output: None,
        }
    }
}

impl Function for Tanh {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![(x.exp() - x.exp().powi(-1)) / (x.exp() + x.exp().powi(-1))]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(v) = &self.output {
            let y = v.clone();
            let gy = gys[0].clone();
            return vec![gy * (1.0 - y.powi(2))];
        } else {
            return vec![];
        }
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Reshape {
    shape: Vec<usize>,
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
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

impl Function for Reshape {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        let reshaped = x.to_shape(self.shape.clone());
        vec![reshaped.unwrap().to_owned()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.output {
            let gy = gys[0].clone();
            let shape = self
                .input
                .clone()
                .unwrap()
                .content
                .borrow()
                .data
                .shape()
                .to_vec();
            return vec![reshape(gy, shape)];
        } else {
            return vec![];
        }
    }
}

pub fn reshape(x: Variable, shape: Vec<usize>) -> Variable {
    if x.data().shape() == shape {
        x
    } else {
        Reshape::new(shape)(x)
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Transpose {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Transpose {
    pub fn new() -> Self {
        Transpose {
            input: None,
            output: None,
        }
    }
}

impl Function for Transpose {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        vec![x.t().to_owned()]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        if let Some(_v) = &self.output {
            let gy = gys[0].clone();
            return vec![transpose(gy)];
        } else {
            return vec![];
        }
    }
}

pub fn transpose(x: Variable) -> Variable {
    Transpose::new()(x)
}

#[derive(UniFunction, FunctionNode)]
pub struct Sum {
    axis: usize,
    keep_dims: bool,
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl Sum {
    pub fn new() -> Self {
        Sum {
            axis: usize::MAX,
            keep_dims: false,
            input: None,
            output: None,
        }
    }

    pub fn new_axis_keep_dim(axis: usize, keep_dims: bool) -> Self {
        Sum {
            axis,
            keep_dims,
            input: None,
            output: None,
        }
    }
}

impl Function for Sum {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        let x = inputs.get(0).clone().unwrap();
        let y = if self.axis == usize::MAX {
            ndarray::array![x.sum()].into_dyn()
        } else {
            x.sum_axis(ndarray::Axis(self.axis))
        };
        let y = if self.keep_dims {
            y.to_shape(vec![1; x.ndim()]).unwrap().to_owned()
        } else {
            y
        };
        vec![y]
    }

    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        let gy = gys[0].clone();
        let x_dim = self.input.clone().unwrap().content.borrow().data.dim();
        let gx = bloadcast_to(gy, x_dim);
        vec![gx]
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct BloadcastTo {
    dim: Dim<IxDynImpl>,
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl BloadcastTo {
    pub fn new(dim: Dim<IxDynImpl>) -> Self {
        BloadcastTo {
            dim,
            input: None,
            output: None,
        }
    }
}

impl Function for BloadcastTo {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        let x = inputs.get(0).unwrap();
        vec![x.broadcast(self.dim.clone()).unwrap().to_owned()]
    }

    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        let gy = gys[0].clone();
        let x_dim = self.input.clone().unwrap().content.borrow().data.dim();
        vec![sum_to(gy, x_dim)]
    }
}

pub fn bloadcast_to(x: Variable, dim: Dim<IxDynImpl>) -> Variable {
    if x.data().dim() == dim {
        x
    } else {
        BloadcastTo::new(dim)(x)
    }
}

#[derive(UniFunction, FunctionNode)]
struct SumTo {
    dim: Dim<IxDynImpl>,
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
}

impl SumTo {
    pub fn new(dim: Dim<IxDynImpl>) -> Self {
        SumTo {
            dim,
            input: None,
            output: None,
        }
    }
}

impl Function for SumTo {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        let x = inputs.get(0).unwrap();
        let shape = self.dim.slice();
        vec![utils::sum_to(x, shape)]
    }

    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        let gy = gys[0].clone();
        let x_dim = self.input.clone().unwrap().content.borrow().data.dim();
        vec![bloadcast_to(gy, x_dim)]
    }
}

fn sum_to(x: Variable, dim: Dim<IxDynImpl>) -> Variable {
    if x.data().dim() == dim {
        x
    } else {
        SumTo::new(dim)(x)
    }
}

pub fn sigmoid(x: Variable) -> Variable {
    Sigmoid::new()(x)
}

#[cfg(test)]
mod tests {
    use crate::no_grad;
    use ndarray::{Array1, ArrayBase, ArrayD, ArrayView2, OwnedRepr};

    use super::*;

    #[test]
    fn square_test() {
        let x1 = vec![5.0, 10.0];
        let expected: Vec<f64> = x1.iter().map(|&x| x * x).collect();
        let x2 = VarData::new(Array1::from_vec(x1).into_dyn());
        let mut f = Square::new();
        let actual = f.call(&[Variable::from(x2)]);
        assert_eq!(1, actual.len());
        let mut result = Vec::new();
        let var = actual[0].content.borrow_mut();
        for data in var.data.clone() {
            result.push(data);
        }
        assert_eq!(expected, result);
    }

    #[test]
    fn same_var_test() {
        let x = VarData::from_vec1(vec![3.0]).to_node();
        let mut add = Add::new();
        let mut y = add(x.clone(), x.clone());
        y.backward();
        let grad = x.get_grad_vec();
        assert_eq!(vec![2.0], grad)
    }

    #[test]
    fn backward_test() {
        let input = vec![10.0];
        let x = VarData::from_vec1(input.clone()).to_node();
        let expected: Vec<f64> = input.iter().map(|x| 2.0 * x).collect();
        let mut f = Square::new();
        let mut y = f(x.clone());
        y.backward();
        let grad = x.get_grad_vec();
        assert_eq!(expected, grad)
    }

    #[test]
    fn calc_graph_test() {
        let x = VarData::from_vec1(vec![2.0]).to_node();
        let mut square = Square::new();
        let mut add = Add::new();
        let a = square(x.clone());
        let (a1, a2) = (a.clone(), a.clone());
        let a1 = square(a1);
        let a2 = square(a2);
        let mut y = add(a1, a2);
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
            let x2 = VarData::new(Array1::from_vec(x1).into_dyn());
            let mut f = Square::new();
            let actual = f.call(&[Variable::from(x2)]);
            assert_eq!(1, actual.len());
            let mut result = Vec::new();
            let var = actual[0].content.borrow_mut();
            for data in var.data.clone() {
                result.push(data);
            }
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn sum_function_test() {
        let base = ndarray::array![[1., 2., 3.], [4., 5., 6.]];
        let x = Variable::from_arry(base.into_dyn());
        let mut y = Sum::new_axis_keep_dim(0, false)(x.clone());
        y.backward();
        println!("{}", y);
        println!("{}", x.grad().unwrap());
        let x = Variable::from_arry(
            ndarray::array![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]].into_dyn(),
        );
        let mut y = Sum::new_axis_keep_dim(usize::MAX, true).apply(x.clone());
        y.backward();
        println!("{}", y);
        println!("{}", x.grad().unwrap());
    }

    #[test]
    fn sum_to_function_test() {
        let base = ndarray::array![[1., 2., 3.], [4., 5., 6.]];
        let x = Variable::from_arry(base.into_dyn());
        let y = sum_to(x.clone(), IxDyn(&[1, 3]));
        println!("{}", y);
        let y = sum_to(x, IxDyn(&[2, 1]));
        println!("{}", y);
    }

    #[test]
    fn bloadcast_add_test() {
        let x0 = Variable::from_arry(ndarray::array![1., 2., 3.].into_dyn());
        let x1 = Variable::from_arry(ndarray::array![10.0].into_dyn());
        let mut y = x0.clone() + x1.clone();
        println!("{}", y);
        y.backward();
        println!("{}", x0.grad().unwrap());
        println!("{}", x1.grad().unwrap());
    }

    #[test]
    fn matmul_test() {
        let a: ArrayBase<OwnedRepr<f64>, _> =
            ArrayBase::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let x = Variable::from_arry(a);
        let b: ArrayBase<OwnedRepr<f64>, _> =
            ArrayBase::from_shape_vec(IxDyn(&[3, 2]), vec![7., 8., 9., 10., 11., 12.]).unwrap();
        let w = Variable::from_arry(b);
        let mut y = matmal(x.clone(), w.clone());
        y.backward();
        println!("{}", x.grad().unwrap());
        println!("{}", w.grad().unwrap());
    }
}
