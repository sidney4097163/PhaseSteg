@echo off
chcp 65001 >nul
setlocal
REM ======================================================================
REM  PhaseSteg 全部实验（Windows 10 + NVIDIA GPU）
REM  第一阶段（默认）：只训练 5 个关键模型并评估，约 14 小时，跑完自动打包后停止。
REM      PowerShell:  $env:ITERS="";  .\run_all.bat
REM  第二阶段：根据第一阶段结果确定主配置后再运行：
REM      PowerShell:  $env:STAGE="B";  .\run_all.bat
REM  已完成的训练（存在 final.pt）会自动跳过；中断后重跑会从 last.pt 续训。
REM ======================================================================
if "%ITERS%"=="" set ITERS=40000
if "%BATCH%"=="" set BATCH=8
if "%WORKERS%"=="" set WORKERS=4
if "%CS%"=="" set CS=2
set PY=python
set COCO=data\coco
set CELEBA=data\celeba
set COM=--iters %ITERS% --batch %BATCH% --workers %WORKERS%

echo ===== 0. 自检 =====
%PY% crypto_chaos.py || goto :err
%PY% models.py || goto :err

echo ===== 1. 加密层分析 (CPU) =====
if not exist results\cipher\cipher_analysis.json %PY% cipher_analysis.py --test %COCO%\test --n 100 --out results\cipher || goto :err

if "%STAGE%"=="B" goto :stageB
if "%STAGE%"=="C" goto :stageC
if "%STAGE%"=="D" goto :stageD

echo ===== A. 关键模型（COCO）=====
REM  ISGI 复现（卷积自编码器 + 原始判别器 + 线性表示 + 不做量化感知，同原文）
if not exist runs\coco_isgi_lin\final.pt     %PY% train.py --name coco_isgi_lin     --data %COCO%\train --val %COCO%\test --arch isgi --rep lin   --disc isgi --quant 0 %COM% || goto :err
REM  本文：相位 + HINN，无冗余 / 2x2 密文域冗余
if not exist runs\coco_hinn_phase\final.pt   %PY% train.py --name coco_hinn_phase   --data %COCO%\train --val %COCO%\test --arch hinn --rep phase --disc srm %COM% || goto :err
if not exist runs\coco_hinn_phase_s2\final.pt %PY% train.py --name coco_hinn_phase_s2 --data %COCO%\train --val %COCO%\test --arch hinn --rep phase --disc srm --cscale 2 %COM% || goto :err
REM  线性表示对照
if not exist runs\coco_hinn_lin_s2\final.pt  %PY% train.py --name coco_hinn_lin_s2  --data %COCO%\train --val %COCO%\test --arch hinn --rep lin   --disc srm --cscale 2 %COM% || goto :err
REM  ISGI 网络 + 2x2 冗余（输入熵降到 1/4，验证瓶颈分析）
if not exist runs\coco_isgi_lin_s2\final.pt  %PY% train.py --name coco_isgi_lin_s2  --data %COCO%\train --val %COCO%\test --arch isgi --rep lin   --disc isgi --quant 0 --cscale 2 %COM% || goto :err

for %%R in (coco_isgi_lin coco_hinn_phase coco_hinn_phase_s2 coco_hinn_lin_s2 coco_isgi_lin_s2) do (
  if not exist runs\%%R\eval_coco.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 1000 || goto :err
)
%PY% collect_results.py
echo.
echo 第一阶段完成。请把 results_pack.zip 发回来，确定主配置后再运行第二阶段。
goto :eof

:stageB
echo ===== B1. 其余主实验与参考上界（COCO）=====
if not exist runs\coco_isgi_phase\final.pt %PY% train.py --name coco_isgi_phase --data %COCO%\train --val %COCO%\test --arch isgi --rep phase --disc isgi --quant 0 %COM% || goto :err
if not exist runs\coco_hinn_plain\final.pt %PY% train.py --name coco_hinn_plain --data %COCO%\train --val %COCO%\test --arch hinn --rep plain --disc srm %COM% || goto :err
if not exist runs\coco_isgi_plain\final.pt %PY% train.py --name coco_isgi_plain --data %COCO%\train --val %COCO%\test --arch isgi --rep plain --disc isgi --quant 0 %COM% || goto :err

echo ===== B2. 消融（HINN + phase, cscale=%CS%）=====
set ABL=--data %COCO%\train --val %COCO%\test --arch hinn --rep phase --cscale %CS%
if not exist runs\abl_noquant\final.pt  %PY% train.py --name abl_noquant  %ABL% --disc srm --quant 0 %COM% || goto :err
if not exist runs\abl_nodisc\final.pt   %PY% train.py --name abl_nodisc   %ABL% --disc none %COM% || goto :err
if not exist runs\abl_isgidisc\final.pt %PY% train.py --name abl_isgidisc %ABL% --disc isgi %COM% || goto :err
if not exist runs\abl_nolow\final.pt    %PY% train.py --name abl_nolow    %ABL% --disc srm --lam_low 0 %COM% || goto :err
if not exist runs\abl_b4\final.pt       %PY% train.py --name abl_b4       %ABL% --disc srm --blocks 4 %COM% || goto :err
REM  随机字节训练 vs 真实 CPCE 密文训练（各 10000 次迭代，等长对比）
if not exist runs\abl_rand10k\final.pt  %PY% train.py --name abl_rand10k  %ABL% --disc srm %COM% --iters 10000 || goto :err
if not exist runs\abl_real10k\final.pt  %PY% train.py --name abl_real10k  %ABL% --disc srm --train_cipher cpce %COM% --iters 10000 || goto :err
REM  鲁棒版本（高斯 / JPEG 噪声层）
if not exist runs\coco_main_robust\final.pt %PY% train.py --name coco_main_robust %ABL% --disc srm --noise 1 %COM% || goto :err

echo ===== B3. CelebA =====
if not exist runs\celeba_isgi_lin\final.pt   %PY% train.py --name celeba_isgi_lin   --data %CELEBA%\train --val %CELEBA%\test --arch isgi --rep lin   --disc isgi --quant 0 %COM% || goto :err
if not exist runs\celeba_main\final.pt       %PY% train.py --name celeba_main       --data %CELEBA%\train --val %CELEBA%\test --arch hinn --rep phase --disc srm --cscale %CS% %COM% || goto :err
if not exist runs\celeba_hinn_plain\final.pt %PY% train.py --name celeba_hinn_plain --data %CELEBA%\train --val %CELEBA%\test --arch hinn --rep plain --disc srm %COM% || goto :err

echo ===== B4. 评估（1000 对，8 bit 含密图，解密域指标）=====
for %%R in (coco_isgi_phase coco_hinn_plain coco_isgi_plain abl_noquant abl_nodisc abl_isgidisc abl_nolow abl_b4 abl_rand10k abl_real10k coco_main_robust) do (
  if not exist runs\%%R\eval_coco.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 1000 || goto :err
)
for %%R in (celeba_isgi_lin celeba_main celeba_hinn_plain) do (
  if not exist runs\%%R\eval_celeba.json %PY% evaluate.py --run runs\%%R --test %CELEBA%\test --n_pairs 1000 || goto :err
)
set MAIN=coco_hinn_phase_s2
if "%CS%"=="1" set MAIN=coco_hinn_phase
REM  跨数据集：COCO 训练 -> CelebA 测试
if not exist runs\%MAIN%\eval_celeba_cross.json  %PY% evaluate.py --run runs\%MAIN%       --test %CELEBA%\test --n_pairs 500 --ciphers cpce --tag cross || goto :err
if not exist runs\coco_isgi_lin\eval_celeba_cross.json %PY% evaluate.py --run runs\coco_isgi_lin --test %CELEBA%\test --n_pairs 500 --ciphers ieps,cpce --tag cross || goto :err

REM  保护带宽度消融（只需评估，不需重训）
if not exist runs\%MAIN%\eval_coco_guard.json %PY% evaluate.py --run runs\%MAIN% --test %COCO%\test --n_pairs 500 --ciphers cpce0,cpce4,cpce8,cpce,cpce24,cpce32,cpce48 --tag guard || goto :err

echo ===== B5. 鲁棒性 =====
for %%R in (coco_isgi_lin_s2 %MAIN% coco_main_robust) do (
  if not exist runs\%%R\eval_coco_robust.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 300 --ciphers cpce --attacks none,gauss1,gauss2,jpeg95,jpeg90,jpeg75 --tag robust || goto :err
)

echo ===== B6. 深度隐写分析 SRNet（2000 对：60/10/30 划分）=====
if not exist stego\isgi_lin_ieps\cover %PY% evaluate.py --run runs\coco_isgi_lin --test %COCO%\test --n_pairs 2000 --ciphers ieps --tag sa --save_stego stego\isgi_lin_ieps || goto :err
if not exist stego\main_cpce\cover     %PY% evaluate.py --run runs\%MAIN%        --test %COCO%\test --n_pairs 2000 --ciphers cpce --tag sa --save_stego stego\main_cpce || goto :err
if not exist stego\nodisc_cpce\cover   %PY% evaluate.py --run runs\abl_nodisc    --test %COCO%\test --n_pairs 2000 --ciphers cpce --tag sa --save_stego stego\nodisc_cpce || goto :err
if not exist stego\isgidisc_cpce\cover %PY% evaluate.py --run runs\abl_isgidisc  --test %COCO%\test --n_pairs 2000 --ciphers cpce --tag sa --save_stego stego\isgidisc_cpce || goto :err
if not exist stego\hinn_plain\cover    %PY% evaluate.py --run runs\coco_hinn_plain --test %COCO%\test --n_pairs 2000 --tag sa --save_stego stego\hinn_plain || goto :err
for %%S in (isgi_lin_ieps main_cpce nodisc_cpce isgidisc_cpce hinn_plain) do (
  if not exist stego\%%S\steganalysis_srnet.json %PY% steganalysis.py --dir stego\%%S --epochs 60 --workers %WORKERS% || goto :err
)

echo ===== B7. 汇总打包 =====
%PY% collect_results.py
echo 全部完成，请把 results_pack.zip 发回来。
goto :eof

:stageC
echo ===== C. 无判别器的主配置与对照（主模型 = abl_nodisc）=====
set NC=--disc none %COM%
set CO=--data %COCO%\train --val %COCO%\test --arch hinn
if not exist runs\c_s1_phase\final.pt %PY% train.py --name c_s1_phase %CO% --rep phase %NC% || goto :err
if not exist runs\c_s2_lin\final.pt   %PY% train.py --name c_s2_lin   %CO% --rep lin --cscale 2 %NC% || goto :err
if not exist runs\c_noquant\final.pt  %PY% train.py --name c_noquant  %CO% --rep phase --cscale 2 --quant 0 %NC% || goto :err
if not exist runs\c_nolow\final.pt    %PY% train.py --name c_nolow    %CO% --rep phase --cscale 2 --lam_low 0 %NC% || goto :err
if not exist runs\c_b4\final.pt       %PY% train.py --name c_b4       %CO% --rep phase --cscale 2 --blocks 4 %NC% || goto :err
if not exist runs\c_coco_plain\final.pt %PY% train.py --name c_coco_plain %CO% --rep plain %NC% || goto :err
if not exist runs\c_robust\final.pt   %PY% train.py --name c_robust   %CO% --rep phase --cscale 2 --noise 1 %NC% || goto :err
if not exist runs\c_celeba_main\final.pt  %PY% train.py --name c_celeba_main  --data %CELEBA%\train --val %CELEBA%\test --arch hinn --rep phase --cscale 2 %NC% || goto :err
if not exist runs\c_celeba_plain\final.pt %PY% train.py --name c_celeba_plain --data %CELEBA%\train --val %CELEBA%\test --arch hinn --rep plain %NC% || goto :err

echo ===== C 评估 =====
for %%R in (c_s1_phase c_s2_lin c_noquant c_nolow c_b4 c_coco_plain c_robust) do (
  if not exist runs\%%R\eval_coco.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 1000 || goto :err
)
for %%R in (c_celeba_main c_celeba_plain) do (
  if not exist runs\%%R\eval_celeba.json %PY% evaluate.py --run runs\%%R --test %CELEBA%\test --n_pairs 1000 || goto :err
)
if not exist runs\abl_nodisc\eval_coco_guard.json  %PY% evaluate.py --run runs\abl_nodisc --test %COCO%\test --n_pairs 500 --ciphers cpce0,cpce4,cpce8,cpce,cpce24,cpce32,cpce48 --tag guard || goto :err
if not exist runs\abl_nodisc\eval_celeba_cross.json %PY% evaluate.py --run runs\abl_nodisc --test %CELEBA%\test --n_pairs 500 --ciphers cpce --tag cross || goto :err
for %%R in (abl_nodisc c_robust) do (
  if not exist runs\%%R\eval_coco_robust.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 300 --ciphers cpce --attacks none,gauss1,gauss2,jpeg95,jpeg90,jpeg75 --tag robust || goto :err
)
if not exist runs\abl_nodisc\vis_coco_cpce.jpg %PY% evaluate.py --run runs\abl_nodisc --test %COCO%\test --n_pairs 8 --ciphers cpce,ieps --tag vis || goto :err
%PY% collect_results.py
echo 第三阶段完成，请把 results_pack.zip 发回来。
goto :eof

:stageD
echo ===== D. SCI 补充实验 =====
echo ----- D0. 加密层分析（含 SHAKE-256 标准流密码对照，CPU）-----
if not exist results\cipher_v2\cipher_analysis.json %PY% cipher_analysis.py --test %COCO%\test --n 100 --out results\cipher_v2 || goto :err

echo ----- D1. 随机种子：主方法与线性表示各补 2 个种子 -----
set CO=--data %COCO%\train --val %COCO%\test --disc none %COM%
if not exist runs\d_main_seed1\final.pt %PY% train.py --name d_main_seed1 %CO% --arch hinn --rep phase --cscale 2 --seed 1 || goto :err
if not exist runs\d_main_seed2\final.pt %PY% train.py --name d_main_seed2 %CO% --arch hinn --rep phase --cscale 2 --seed 2 || goto :err
if not exist runs\d_lin_seed1\final.pt  %PY% train.py --name d_lin_seed1  %CO% --arch hinn --rep lin --cscale 2 --seed 1 || goto :err
if not exist runs\d_lin_seed2\final.pt  %PY% train.py --name d_lin_seed2  %CO% --arch hinn --rep lin --cscale 2 --seed 2 || goto :err

echo ----- D2. 第二个基线：Baluja 网络（异或加密流程 / 接入本文 CPCE+相位+冗余）-----
if not exist runs\d_bal_xor\final.pt  %PY% train.py --name d_bal_xor  %CO% --arch baluja --rep lin || goto :err
if not exist runs\d_bal_main\final.pt %PY% train.py --name d_bal_main %CO% --arch baluja --rep phase --cscale 2 || goto :err

echo ----- D3. 评估 -----
for %%R in (d_main_seed1 d_main_seed2 d_lin_seed1 d_lin_seed2) do (
  if not exist runs\%%R\eval_coco.json %PY% evaluate.py --run runs\%%R --test %COCO%\test --n_pairs 1000 || goto :err
)
if not exist runs\d_bal_xor\eval_coco.json  %PY% evaluate.py --run runs\d_bal_xor  --test %COCO%\test --n_pairs 1000 --ciphers xor,ieps,cpce || goto :err
if not exist runs\d_bal_main\eval_coco.json %PY% evaluate.py --run runs\d_bal_main --test %COCO%\test --n_pairs 1000 --ciphers cpce,cpce0,xor,ieps || goto :err
REM  标准流密码对照：同一主模型换用 SHAKE-256 模加 / 异或（不需重训）
if not exist runs\abl_nodisc\eval_coco_std.json %PY% evaluate.py --run runs\abl_nodisc --test %COCO%\test --n_pairs 1000 --ciphers cpce,shake,xor --tag std || goto :err

echo ----- D4. 第三个数据集 DIV2K（COCO 训练的模型直接测试）-----
if exist data\div2k\test (
  if not exist runs\abl_nodisc\eval_div2k_cross.json    %PY% evaluate.py --run runs\abl_nodisc    --test data\div2k\test --n_pairs 400 --ciphers cpce,ieps --tag cross || goto :err
  if not exist runs\coco_isgi_lin\eval_div2k_cross.json %PY% evaluate.py --run runs\coco_isgi_lin --test data\div2k\test --n_pairs 400 --ciphers ieps,cpce --tag cross || goto :err
  if not exist runs\d_bal_main\eval_div2k_cross.json    %PY% evaluate.py --run runs\d_bal_main    --test data\div2k\test --n_pairs 400 --ciphers cpce --tag cross || goto :err
  if not exist runs\c_coco_plain\eval_div2k_cross.json  %PY% evaluate.py --run runs\c_coco_plain  --test data\div2k\test --n_pairs 400 --tag cross || goto :err
) else (
  echo 未找到 data\div2k\test，跳过 DIV2K 测试
)

%PY% collect_results.py --pack_runs "d_*,abl_nodisc,coco_isgi_lin,c_coco_plain"
echo 第四阶段完成，请把 results_pack.zip 发回来。
goto :eof

:err
echo.
echo *** 出错，已停止。请把上面的报错信息发给我。修复后重新运行 run_all.bat 会跳过已完成的部分。***
exit /b 1
