;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result f64)
        i64.const 1
        f64.convert_i64_u
        block
        end
    )
)
;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    (%r11), %r11
;;       addq    $0x18, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x78
;;   1b: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movq    $1, %rcx
;;       cmpq    $0, %rcx
;;       jl      0x46
;;   3c: cvtsi2sdq %rcx, %xmm0
;;       jmp     0x60
;;   46: movq    %rcx, %r11
;;       shrq    $1, %r11
;;       movq    %rcx, %rax
;;       andq    $1, %rax
;;       orq     %r11, %rax
;;       cvtsi2sdq %rax, %xmm0
;;       addsd   %xmm0, %xmm0
;;       subq    $8, %rsp
;;       movsd   %xmm0, (%rsp)
;;       movsd   (%rsp), %xmm0
;;       addq    $8, %rsp
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   78: ud2