from math import isclose
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = ROOT.joinpath('index.html').resolve().as_uri()

QWEN_MODEL_INFO = {
    'modelId': 'Qwen/Qwen3-32B',
    'tags': ['transformers', 'safetensors', 'qwen3', 'text-generation'],
    'config': {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3'},
}
QWEN_CONFIG = {
    'architectures': ['Qwen3ForCausalLM'],
    'model_type': 'qwen3',
    'hidden_size': 5120,
    'num_hidden_layers': 64,
    'num_attention_heads': 64,
    'num_key_value_heads': 8,
    'head_dim': 128,
    'max_position_embeddings': 40960,
    'torch_dtype': 'bfloat16',
}
QWEN_INDEX = {'metadata': {'total_size': 65524246528}}

DEEPSEEK_MODEL_INFO = {
    'modelId': 'deepseek-ai/DeepSeek-R1',
    'tags': ['transformers', 'safetensors', 'deepseek_v3', 'text-generation', 'fp8'],
    'safetensors': {
        'parameters': {
            'BF16': 3918786560,
            'F8_E4M3': 680571043840,
            'F32': 41555600,
        }
    },
    'config': {
        'architectures': ['DeepseekV3ForCausalLM'],
        'model_type': 'deepseek_v3',
        'quantization_config': {'quant_method': 'fp8'},
    },
}
DEEPSEEK_CONFIG = {
    'architectures': ['DeepseekV3ForCausalLM'],
    'model_type': 'deepseek_v3',
    'hidden_size': 7168,
    'num_hidden_layers': 61,
    'num_attention_heads': 128,
    'num_key_value_heads': 128,
    'max_position_embeddings': 163840,
    'kv_lora_rank': 512,
    'qk_rope_head_dim': 64,
    'qk_nope_head_dim': 128,
    'v_head_dim': 128,
    'n_routed_experts': 256,
    'num_experts_per_tok': 8,
    'torch_dtype': 'bfloat16',
    'quantization_config': {'quant_method': 'fp8'},
}


def install_routes(page):
    def handler(route):
        url = route.request.url
        if '/api/models/Qwen/Qwen3-32B' in url:
            route.fulfill(status=200, content_type='application/json', body=__import__('json').dumps(QWEN_MODEL_INFO))
            return
        if '/Qwen/Qwen3-32B/resolve/main/config.json' in url:
            route.fulfill(status=200, content_type='application/json', body=__import__('json').dumps(QWEN_CONFIG))
            return
        if '/Qwen/Qwen3-32B/resolve/main/model.safetensors.index.json' in url:
            route.fulfill(status=200, content_type='application/json', body=__import__('json').dumps(QWEN_INDEX))
            return
        if '/api/models/deepseek-ai/DeepSeek-R1' in url:
            route.fulfill(status=200, content_type='application/json', body=__import__('json').dumps(DEEPSEEK_MODEL_INFO))
            return
        if '/deepseek-ai/DeepSeek-R1/resolve/main/config.json' in url:
            route.fulfill(status=200, content_type='application/json', body=__import__('json').dumps(DEEPSEEK_CONFIG))
            return
        if '/deepseek-ai/DeepSeek-R1/resolve/main/model.safetensors.index.json' in url:
            route.fulfill(status=404, content_type='application/json', body='{}')
            return
        route.continue_()

    page.route('https://huggingface.co/**', handler)


def expect_close(actual, expected, tolerance=0.03, label='value'):
    if not isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f'{label}: expected {expected}, got {actual}')


def test_qwen_hidden_size_and_manual_quantization(page):
    page.goto(URL)
    page.wait_for_selector('#hf-model-id')
    page.fill('#hf-model-id', 'Qwen/Qwen3-32B')
    page.click('#hf-fetch-btn')
    page.wait_for_function("document.querySelector('#hf-status').textContent.includes('成功加载')")

    overhead_value = float(page.input_value('#overhead-estimate'))
    expect_close(overhead_value, 0.67, label='Qwen overhead estimate')

    breakdown_rows = page.locator('#breakdown-body tr').all_inner_texts()
    if 'HF model.safetensors.index.json' not in breakdown_rows[1]:
        raise AssertionError('Qwen should use checkpoint-derived weights when quantization matches fetched model')

    page.select_option('#quant-method', 'int4')
    page.wait_for_timeout(100)

    breakdown_rows = page.locator('#breakdown-body tr').all_inner_texts()
    if '参数量：32.0B × 0.5313 bytes' not in breakdown_rows[1]:
        raise AssertionError('Qwen int4 override should fall back to parameter-based estimation with metadata overhead')
    if '8.50 GB' not in breakdown_rows[0]:
        raise AssertionError('Qwen int4 per-GPU weight should reflect metadata-aware estimate')


def test_deepseek_checkpoint_source_and_mla(page):
    page.goto(URL)
    page.wait_for_selector('#hf-model-id')
    page.fill('#gpu-vram', '80')
    page.fill('#num-gpus', '16')
    page.fill('#hf-model-id', 'deepseek-ai/DeepSeek-R1')
    page.click('#hf-fetch-btn')
    page.wait_for_function("document.querySelector('#hf-status').textContent.includes('成功加载')")

    quant_method = page.locator('#quant-method').input_value()
    if quant_method != 'fp8':
        raise AssertionError(f'DeepSeek quant method should auto-detect to fp8, got {quant_method}')

    formula_text = page.locator('#bytes-per-token-formula').inner_text()
    if '512 kv_lora_rank' not in formula_text:
        raise AssertionError('DeepSeek should use MLA kv_lora_rank from fetched config')

    breakdown_rows = page.locator('#breakdown-body tr').all_inner_texts()
    if 'HF safetensors.parameters' not in breakdown_rows[1]:
        raise AssertionError('DeepSeek should prefer dtype-aware safetensors.parameters weight sizing')
    if '43.04 GB' not in breakdown_rows[0]:
        raise AssertionError('DeepSeek per-GPU weight should use dtype-aware checkpoint sizing')


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        install_routes(page)
        test_qwen_hidden_size_and_manual_quantization(page)
        page.context.clear_cookies()
        page.evaluate('localStorage.clear()')
        test_deepseek_checkpoint_source_and_mla(page)
        browser.close()
        print('web accuracy regressions passed')


if __name__ == '__main__':
    main()
