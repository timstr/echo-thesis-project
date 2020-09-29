@echo off

SET /A numreceivers=4
SET /A arrangement=1
SET arrangement_str=flat

:beginloop
if %numreceivers% LEQ 16 (
    
    if %arrangement% EQU 0 (
        set arrangement_str=flat
        set /A arrangement=1
    ) else (
        set arrangement_str=grid
        set /A arrangement=0
        set /A numreceivers=%numreceivers% * 2
    )

    echo Running receiver study with %numreceivers% receivers in %arrangement_str% formation
    python echolearn.py --experiment=receiver_study_%numreceivers%_%arrangement_str% --dataset=v8 --batchsize=4 --receivers=%numreceivers% --arrangement=%arrangement_str% --maxobstacles=2 --iterations=30000

    goto beginloop
)
