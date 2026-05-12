# validate_sota — best-effort grid

- checkpoint: `/home/kondrashov_k/mipt/hw/nlp/spacebyte/spacebyte-200M-v2/Train--batch_size=64--beta2=0.98--context_size=2048--d_local=384--d_model=1024--dataset=pg19--device=cuda:0--global_context_size=1024--iters=3e9/tokens--local_attention_window=384--lr=0.5e-2*B**0.5--micro_batch_size=4--model=SpaceByte--n_layers=12--n_local_layers=12--out_dir=spacebyte-200M-v2--patch_method=utf8--rope=True/ckpt_best_loss.pt`
- router: `/home/kondrashov_k/mipt/checkpoint_best_router.pt`
- eval_iters: 50

| split | mode | no_global_routing | no_window_routing | threshold | baseline_ce | baseline_bpb | baseline_real_flops_b | method_ce | method_bpb | method_real_flops_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | full_router | false | false | 0.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | full_router | false | false | 0.1 | 0.288613 | 0.416381 | 112090521.600000 | 0.345000 | 0.497730 | 102349578.240000 |
| val | full_router | false | false | 0.2 | 0.288613 | 0.416381 | 112090521.600000 | 0.353047 | 0.509339 | 98301665.280000 |
| val | full_router | false | false | 0.3 | 0.288613 | 0.416381 | 112090521.600000 | 0.367148 | 0.529683 | 91843584.000000 |
| val | full_router | false | false | 0.4 | 0.288613 | 0.416381 | 112090521.600000 | 0.384141 | 0.554198 | 83690127.360000 |
| val | full_router | false | false | 0.5 | 0.288613 | 0.416381 | 112090521.600000 | 0.409453 | 0.590716 | 71875829.760000 |
| val | full_router | false | false | 0.6 | 0.288613 | 0.416381 | 112090521.600000 | 0.435586 | 0.628418 | 59600486.400000 |
| val | full_router | false | false | 0.7 | 0.288613 | 0.416381 | 112090521.600000 | 0.458516 | 0.661498 | 48936222.720000 |
| val | full_router | false | false | 0.8 | 0.288613 | 0.416381 | 112090521.600000 | 0.468711 | 0.676207 | 44311142.400000 |
| val | full_router | false | false | 0.9 | 0.288613 | 0.416381 | 112090521.600000 | 0.470039 | 0.678123 | 43696128.000000 |
| val | full_router | false | false | 1.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.470039 | 0.678123 | 43696128.000000 |
| val | router_global_only | false | true | 0.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | router_global_only | false | true | 0.1 | 0.288613 | 0.416381 | 112090521.600000 | 0.295566 | 0.426412 | 108395274.240000 |
| val | router_global_only | false | true | 0.2 | 0.288613 | 0.416381 | 112090521.600000 | 0.303379 | 0.437683 | 104347361.280000 |
| val | router_global_only | false | true | 0.3 | 0.288613 | 0.416381 | 112090521.600000 | 0.316484 | 0.456590 | 97889280.000000 |
| val | router_global_only | false | true | 0.4 | 0.288613 | 0.416381 | 112090521.600000 | 0.332617 | 0.479865 | 89735823.360000 |
| val | router_global_only | false | true | 0.5 | 0.288613 | 0.416381 | 112090521.600000 | 0.356094 | 0.513735 | 77921525.760000 |
| val | router_global_only | false | true | 0.6 | 0.288613 | 0.416381 | 112090521.600000 | 0.380664 | 0.549182 | 65646182.400000 |
| val | router_global_only | false | true | 0.7 | 0.288613 | 0.416381 | 112090521.600000 | 0.401953 | 0.579896 | 54981918.720000 |
| val | router_global_only | false | true | 0.8 | 0.288613 | 0.416381 | 112090521.600000 | 0.411094 | 0.593083 | 50356838.400000 |
| val | router_global_only | false | true | 0.9 | 0.288613 | 0.416381 | 112090521.600000 | 0.412383 | 0.594943 | 49741824.000000 |
| val | router_global_only | false | true | 1.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.412383 | 0.594943 | 49741824.000000 |
| val | router_window_only | true | false | 0.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.1 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.2 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.3 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.4 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.5 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.6 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.7 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.8 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 0.9 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | router_window_only | true | false | 1.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.337383 | 0.486741 | 106044825.600000 |
| val | spacebyte_default | true | true | 0.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.1 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.2 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.3 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.4 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.5 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.6 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.7 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.8 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 0.9 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
| val | spacebyte_default | true | true | 1.0 | 0.288613 | 0.416381 | 112090521.600000 | 0.288613 | 0.416381 | 112090521.600000 |
