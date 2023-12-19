cd .. && python3 run_eval.py \
    --model_id="./cn_model_out_trpro" \
    --dataset="mozilla-foundation/common_voice_11_0" \
    --config="zh-CN" --device=0 --language="zh" \
    --streaming=False