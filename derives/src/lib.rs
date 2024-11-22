extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(FunctionNode, attributes(node_I, node_O))]
pub fn function_node_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let mut input_idents = Vec::new();
    let mut output_idents = Vec::new();
    let mut else_quote = Vec::new();

    match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => {
                fields.named.into_iter().for_each(|f| {
                    if f.attrs.iter().any(|attr| attr.path().is_ident("node_I")) {
                        input_idents.push(f.ident.unwrap());
                    } else if f.attrs.iter().any(|attr| attr.path().is_ident("node_O")) {
                        output_idents.push(f.ident.unwrap());
                    } else {
                        let ident = f.ident;
                        else_quote.push(quote! {
                            #ident: self.#ident.clone(),
                        });
                    }
                });
            }
            _ => panic!("no unnamed fields are allowed"),
        },
        _ => panic!("expects struct"),
    }

    if input_idents.len() < 1 {
        panic!("input attribute size must be greater than 1");
    }

    if output_idents.len() != 1 {
        panic!("only supported output size is 1");
    }

    let mut input_quote = Vec::new();
    let mut input_append = Vec::new();
    input_idents.into_iter().enumerate().for_each(|(i, ident)| {
        let declare = quote! {
            #ident: Some(inputs[#i].clone()),
        };
        input_quote.push(declare);
        let append = quote! {
            inputs.push(self.#ident.clone().unwrap().clone());
        };
        input_append.push(append);
    });

    let mut output_quote = Vec::new();
    let mut output_append = Vec::new();
    output_idents
        .into_iter()
        .enumerate()
        .for_each(|(i, ident)| {
            let declare = quote! {
                #ident: Some(outputs[#i].clone()),
            };
            output_quote.push(declare);
            let append = quote! {
                outputs.push(self.#ident.clone().unwrap().clone());
            };
            output_append.push(append);
        });

    let expanded = quote! {
        impl FunctionNode for #name {
            fn new_instance(
                &self,
                inputs: &[Variable],
                outputs: &[Variable],
            ) -> Rc<dyn Function> {
                let f = #name {
                    #(#input_quote)*
                    #(#output_quote)*
                    #(#else_quote)*
                };
                Rc::new(f)
            }
            fn get_inputs(&self) -> Vec<Variable> {
                let mut inputs = Vec::new();
                #(#input_append)*
                inputs
            }
            fn get_outputs(&self) -> Vec<Variable> {
                let mut outputs = Vec::new();
                #(#output_append)*
                outputs
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(BiFunction)]
pub fn bifunction_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let target = quote! { crate::core::function::BiFunction };
    let arg = quote! { crate::core::variable::Variable };
    // Build the output, possibly using quasi-quotation
    let expanded = quote! {
        impl #target for #name {}

        impl FnOnce<(#arg, #arg)> for #name {
            type Output = #arg;

            extern "rust-call" fn call_once(mut self, args: (#arg, #arg)) -> Self::Output {
                self.call_mut(args)
            }
        }

        impl FnMut<(#arg, #arg)> for #name {
            extern "rust-call" fn call_mut(&mut self, args: (#arg, #arg)) -> Self::Output {
                self.apply(args.0, args.1)
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(UniFunction)]
pub fn unifunction_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let target = quote! { crate::core::function::UniFunction };
    let arg = quote! { crate::core::variable::Variable };

    let expanded = quote! {
        impl #target for #name {}

        impl FnOnce<(#arg,)> for #name {
            type Output = #arg;

            extern "rust-call" fn call_once(mut self, args: (#arg,)) -> Self::Output {
                self.call_mut(args)
            }
        }

        impl FnMut<(#arg,)> for #name {
            extern "rust-call" fn call_mut(&mut self, args: (#arg,)) -> Self::Output {
                self.apply(args.0)
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(Module)]
pub fn nn_module_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let target = quote! { crate::nn::Module };
    let expanded = quote! {
        impl #target for #name {}
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(Learnable, attributes(learnable))]
pub fn learnable_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let statesments = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => {
                let mut statements = Vec::new();
                for f in fields.named.into_iter() {
                    if let syn::Type::Path(tp) = &f.ty {
                        if tp.path.is_ident("Variable") {
                            let ident = &f.ident;
                            let state = quote! {
                                set.insert(self.#ident.clone());
                            };
                            statements.push(state);
                        }
                    }
                    if f.attrs.iter().any(|attr| attr.path().is_ident("learnable")) {
                        let ident = &f.ident;
                        let state = quote! {
                            for item in self.#ident.parameters() {
                                set.insert(item);
                            }
                        };
                        statements.push(state);
                    }
                }
                statements
            }
            _ => panic!("no unnamed fields are allowed"),
        },
        _ => panic!("expects struct"),
    };

    let expanded = quote! {
        impl Learnable for #name {
            fn parameters(&self) -> std::collections::HashSet<Variable> {
                let mut set = std::collections::HashSet::new();
                #(#statesments)*
                set
            }
        }
    };

    TokenStream::from(expanded)
}
