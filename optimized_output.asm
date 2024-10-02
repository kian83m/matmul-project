	.file	"dgemm_mine.c"
	.text
	.p2align 6
	.globl	square_dgemm
	.type	square_dgemm, @function
square_dgemm:
.LFB0:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$48, %rsp
	.cfi_def_cfa_offset 104
	movl	%edi, -116(%rsp)
	movq	%rcx, -64(%rsp)
	testl	%edi, %edi
	jle	.L1
	movl	%edi, %ebx
	movq	%rdx, %rax
	movl	$0, -44(%rsp)
	movq	%rsi, %r15
	movl	%ebx, %r11d
	sall	$6, %edi
	xorl	%esi, %esi
	sall	$7, %r11d
	movslq	%r11d, %rdx
	leaq	0(,%rdx,8), %rbp
	movslq	%ebx, %rdx
	leaq	8(%rax), %rbx
	leaq	0(,%rdx,8), %r10
	movq	%r10, %r14
.L20:
	movl	%esi, %r9d
	movl	-116(%rsp), %ecx
	addl	$64, %esi
	movslq	-44(%rsp), %r8
	leal	63(%r9), %eax
	movl	$0, -48(%rsp)
	cmpl	%eax, %ecx
	movq	-64(%rsp), %rax
	movl	%esi, -8(%rsp)
	cmovg	%esi, %ecx
	movq	%r15, %rsi
	movq	%r15, -72(%rsp)
	movq	%r14, %r15
	leaq	(%rax,%r8,8), %rax
	movq	%rsi, %r14
	movl	%ecx, -76(%rsp)
	movl	%r9d, %ecx
	movq	%rax, -40(%rsp)
	xorl	%eax, %eax
.L19:
	movl	-116(%rsp), %r10d
	leal	127(%rax), %esi
	movl	%eax, -20(%rsp)
	movq	$0, -104(%rsp)
	movl	%r10d, %edx
	movq	%r8, -16(%rsp)
	subl	%eax, %edx
	cmpl	%esi, %r10d
	movl	$128, %esi
	movl	%r11d, -4(%rsp)
	cmovg	%esi, %edx
	movq	%rbp, (%rsp)
	movq	-40(%rsp), %r10
	movq	%rax, 8(%rsp)
	leal	(%rdx,%rax), %esi
	subl	$1, %edx
	movq	%rbx, 16(%rsp)
	movl	%esi, -24(%rsp)
	leaq	(%r8,%rax), %rsi
	addq	%rdx, %rsi
	notq	%rdx
	leaq	(%rbx,%rsi,8), %rsi
	movq	%rsi, -32(%rsp)
	leaq	0(,%rdx,8), %rsi
	movq	%rsi, -56(%rsp)
.L18:
	movq	-104(%rsp), %rax
	movl	-116(%rsp), %esi
	leal	31(%rax), %edx
	movl	%esi, %r12d
	movl	%eax, %ebx
	subl	%eax, %r12d
	cmpl	%edx, %esi
	movl	$32, %esi
	cmovg	%esi, %r12d
	movl	-76(%rsp), %esi
	cmpl	%esi, %ecx
	jge	.L6
	movl	-20(%rsp), %edx
	leal	(%r12,%rax), %r8d
	cmpl	%edx, -24(%rsp)
	jle	.L6
	cmpl	%eax, %r8d
	jle	.L6
	movl	%r12d, %ebp
	movl	%ecx, -84(%rsp)
	movq	-40(%rsp), %rdx
	movq	%r10, %rsi
	leaq	0(,%rax,8), %r9
	movl	%r12d, %eax
	movq	%r10, 24(%rsp)
	shrl	%ebp
	andl	$-2, %eax
	movl	%ecx, 32(%rsp)
	movl	-44(%rsp), %r11d
	salq	$4, %rbp
	movl	%eax, -88(%rsp)
	addl	%eax, %ebx
	leaq	8(%r9), %rax
	movq	-32(%rsp), %r13
	movq	%r9, -96(%rsp)
	movq	%rax, -112(%rsp)
	movl	%edi, 36(%rsp)
	.p2align 4,,10
	.p2align 3
.L8:
	movq	-56(%rsp), %rax
	movq	-64(%rsp), %rdi
	movl	%r11d, -80(%rsp)
	movq	-72(%rsp), %rcx
	movl	-48(%rsp), %r9d
	leaq	(%rax,%r13), %r10
	leal	(%rbx,%r11), %eax
	cltq
	leaq	(%rdi,%rax,8), %rdi
	.p2align 4,,10
	.p2align 3
.L14:
	movsd	(%r10), %xmm1
	cmpl	$1, %r12d
	je	.L25
	movq	-112(%rsp), %rax
	addq	%rcx, %rax
	cmpq	%rax, %rsi
	jne	.L10
.L25:
	movq	-104(%rsp), %rax
	.p2align 4,,10
	.p2align 3
.L11:
	movsd	(%rcx,%rax,8), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	(%rdx,%rax,8), %xmm0
	movsd	%xmm0, (%rdx,%rax,8)
	addq	$1, %rax
	cmpl	%eax, %r8d
	jg	.L11
	.p2align 4,,10
	.p2align 3
.L17:
	movl	-116(%rsp), %eax
	addq	$8, %r10
	addq	%r15, %rcx
	addl	%eax, %r9d
	cmpq	%r13, %r10
	jne	.L14
	movl	-80(%rsp), %r11d
	movl	%eax, %edi
	addl	$1, -84(%rsp)
	addq	%r15, %r13
	movl	-84(%rsp), %eax
	addq	%r15, %rdx
	addq	%r15, %rsi
	addl	%edi, %r11d
	movl	-76(%rsp), %edi
	cmpl	%edi, %eax
	jne	.L8
	movq	24(%rsp), %r10
	movl	32(%rsp), %ecx
	movl	36(%rsp), %edi
.L6:
	addq	$32, -104(%rsp)
	addq	$256, %r10
	movq	-104(%rsp), %rax
	cmpl	%eax, -116(%rsp)
	jg	.L18
	movq	8(%rsp), %rax
	movl	-4(%rsp), %r11d
	movq	(%rsp), %rbp
	addl	%r11d, -48(%rsp)
	addq	%rbp, -72(%rsp)
	movq	-16(%rsp), %r8
	subq	$-128, %rax
	movq	16(%rsp), %rbx
	cmpl	%eax, -116(%rsp)
	jg	.L19
	movq	%r14, %rax
	addl	%edi, -44(%rsp)
	movq	%r15, %r14
	movl	-8(%rsp), %esi
	movq	%rax, %r15
	cmpl	%esi, -116(%rsp)
	jg	.L20
.L1:
	addq	$48, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L10:
	.cfi_restore_state
	movq	-96(%rsp), %rax
	movapd	%xmm1, %xmm2
	unpcklpd	%xmm2, %xmm2
	leaq	(%rcx,%rax), %r11
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L12:
	movupd	(%r11,%rax), %xmm0
	movupd	(%rsi,%rax), %xmm3
	mulpd	%xmm2, %xmm0
	addpd	%xmm3, %xmm0
	movups	%xmm0, (%rsi,%rax)
	addq	$16, %rax
	cmpq	%rax, %rbp
	jne	.L12
	movl	-88(%rsp), %eax
	cmpl	%eax, %r12d
	je	.L17
	leal	(%r9,%rbx), %eax
	cltq
	mulsd	(%r14,%rax,8), %xmm1
	addsd	(%rdi), %xmm1
	movsd	%xmm1, (%rdi)
	jmp	.L17
	.cfi_endproc
.LFE0:
	.size	square_dgemm, .-square_dgemm
	.globl	dgemm_desc
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"My awesome dgemm."
	.section	.data.rel.local,"aw"
	.align 8
	.type	dgemm_desc, @object
	.size	dgemm_desc, 8
dgemm_desc:
	.quad	.LC0
	.ident	"GCC: (Debian 12.2.0-14) 12.2.0"
	.section	.note.GNU-stack,"",@progbits
