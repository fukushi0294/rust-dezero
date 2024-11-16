extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

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
