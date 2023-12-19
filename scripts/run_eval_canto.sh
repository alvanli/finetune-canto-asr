cd .. && python3 run_eval.py \
    --model_id="./model_out_trpro" \
    --dataset="mozilla-foundation/common_voice_11_0" \
    --config="zh-HK" --device=0 --language="zh" \
    --streaming=False