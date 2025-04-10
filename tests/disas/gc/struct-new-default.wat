;;! target = "x86_64"
;;! flags = "-W function-references,gc"
;;! test = "optimize"

(module
  (type $ty (struct (field (mut f32))
                    (field (mut i8))
                    (field (mut anyref))
                    (field (mut v128))))

  (func (result (ref $ty))
    (struct.new_default $ty)
  )
)
;; function u0:0(i64 vmctx, i64) -> i32 tail {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0+8
;;     gv2 = load.i64 notrap aligned gv1+16
;;     gv3 = vmctx
;;     sig0 = (i64 vmctx, i32, i32, i32, i32) -> i32 tail
;;     fn0 = colocated u1:27 sig0
;;     const0 = 0x00000000000000000000000000000000
;;     stack_limit = gv2
;;
;;                                 block0(v0: i64, v1: i64):
;; @0023                               v9 = iconst.i32 -1342177280
;; @0023                               v4 = iconst.i32 0
;; @0023                               v7 = iconst.i32 48
;; @0023                               v11 = iconst.i32 16
;; @0023                               v12 = call fn0(v0, v9, v4, v7, v11)  ; v9 = -1342177280, v4 = 0, v7 = 48, v11 = 16
;; @0023                               v3 = f32const 0.0
;; @0023                               v14 = load.i64 notrap aligned readonly can_move v0+40
;; @0023                               v15 = uextend.i64 v12
;; @0023                               v16 = iadd v14, v15
;;                                     v49 = iconst.i64 16
;; @0023                               v17 = iadd v16, v49  ; v49 = 16
;; @0023                               store notrap aligned little v3, v17  ; v3 = 0.0
;;                                     v50 = iconst.i64 20
;; @0023                               v18 = iadd v16, v50  ; v50 = 20
;; @0023                               istore8 notrap aligned little v4, v18  ; v4 = 0
;;                                     v52 = iconst.i32 1
;; @0023                               brif v52, block3, block2  ; v52 = 1
;;
;;                                 block2:
;; @0023                               v27 = load.i64 notrap aligned readonly can_move v0+48
;;                                     v78 = iconst.i64 16
;;                                     v79 = icmp uge v27, v78  ; v78 = 16
;; @0023                               trapz v79, user1
;; @0023                               v29 = iconst.i64 8
;; @0023                               v34 = iadd.i64 v14, v29  ; v29 = 8
;; @0023                               v35 = load.i64 notrap aligned v34
;;                                     v54 = iconst.i64 1
;; @0023                               v36 = iadd v35, v54  ; v54 = 1
;; @0023                               store notrap aligned v36, v34
;; @0023                               jump block3
;;
;;                                 block3:
;;                                     v80 = iconst.i32 0
;;                                     v51 = iconst.i64 24
;; @0023                               v19 = iadd.i64 v16, v51  ; v51 = 24
;; @0023                               store notrap aligned little v80, v19  ; v80 = 0
;; @0023                               v6 = vconst.i8x16 const0
;;                                     v55 = iconst.i64 32
;; @0023                               v48 = iadd.i64 v16, v55  ; v55 = 32
;; @0023                               store notrap aligned little v6, v48  ; v6 = const0
;; @0026                               jump block1
;;
;;                                 block1:
;; @0026                               return v12
;; }
