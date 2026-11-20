from pathlib import Path
import ast
p=Path('train.py')
s=p.read_text()
def replace(a,b):
    global s
    if s.count(a)!=1: raise ValueError(f'upstream changed or not unique: {a[:90]}')
    s=s.replace(a,b,1)
replace('import argparse\n','import argparse\nimport time\nfrom fusion_report import write_epoch, write_best\n')
replace("choices=['raw', 'enhanced', 'raw_enhanced']","choices=['A', 'B', 'C']")
replace('or args.preprocess != "crop" or args.resume or args.exist_ok):','or args.preprocess != "crop" or args.resume != args.exist_ok):')
replace('    args = ap.parse_args()\n','    args = ap.parse_args()\n    if args.device != "0" or torch.cuda.device_count()!=1:\n        raise RuntimeError("exactly one visible CUDA device required")\n    torch.cuda.set_per_process_memory_fraction(.12, 0)\n    torch.set_num_threads(1)\n    torch.set_num_interop_threads(1)\n')
replace('    metrics["val_loss"] = val_loss\n    return metrics','    metrics["val_loss"] = val_loss\n    metrics["_fusion_preds"]=(y_true,y_score)\n    return metrics')
replace('    }, path)\n\n\ndef _build_optimizer','    }, path + ".tmp")\n    os.replace(path + ".tmp",path)\n\n\ndef _build_optimizer')
replace('        best_metric = ckpt.get("best_metric", -1.0)\n',
        '        best_metric = ckpt.get("best_metric", -1.0)\n        if os.path.isfile(os.path.join(out_dir,"best.pth")):\n            best_metric=max(best_metric,torch.load(os.path.join(out_dir,"best.pth"),map_location="cpu",weights_only=False)["best_metric"])\n')
replace('    for epoch in range(start_epoch, args.epochs):\n','    for epoch in range(start_epoch, args.epochs):\n        t_epoch=time.monotonic()\n')
replace('        for imgs, targets in train_loader:\n','        for step,(imgs, targets) in enumerate(train_loader,1):\n')
replace('            train_loss += loss.item() * imgs.size(0)\n','            lv=loss.item()\n            if not np.isfinite(lv): raise RuntimeError(f"nonfinite loss: epoch={epoch+1}, step={step}")\n            train_loss += lv * imgs.size(0)\n            if step==1 or step%100==0: print(f"[step] epoch={epoch+1} step={step}/{len(train_loader)} loss={lv:.5f} elapsed={time.monotonic()-t_epoch:.1f}s",flush=True)\n')
replace('        scheduler.step()\n','        y,scores=m.pop("_fusion_preds")\n        scheduler.step()\n')
replace('        save_checkpoint(os.path.join(out_dir, "last.pth"), model, epoch,\n                        best_metric, optimizer, scheduler, m)\n','')
replace('                from fabric_review import write_best\n                write_best(out_dir, model, val_loader, device, names, epoch + 1, amp=args.amp)',
        '                write_best(out_dir,val_ds.stems,names,y,scores,epoch+1,args.fabric_mode,m)')
replace('        # 早停判断: 达到耐心值即停止\n',
        '        save_checkpoint(os.path.join(out_dir,"last.pth"),model,epoch,best_metric,optimizer,scheduler,m)\n        write_epoch(out_dir,names,y,scores,epoch+1,args.fabric_mode,m)\n        print(f"[epoch timing] {epoch+1}/{args.epochs} {time.monotonic()-t_epoch:.1f}s",flush=True)\n\n        # 早停判断: 达到耐心值即停止\n')
ast.parse(s)
p.write_text(s)

