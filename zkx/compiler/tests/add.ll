; Define a function: int add(int a, int b)
define i32 @main(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
