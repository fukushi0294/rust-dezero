extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(BiFunction)]
pub fn bifunction_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let target = quote! { crate::core::function::BiFunction };
    let arg = quote! { crate::core::variable::VarNode };
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
    let arg = quote! { crate::core::variable::VarNode };

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
                        if tp.path.is_ident("VarNode") {
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
            fn parameters(&self) -> std::collections::HashSet<VarNode> {
                let mut set = std::collections::HashSet::new();
                #(#statesments)*
                set
            }
        }
    };

    TokenStream::from(expanded)
}
