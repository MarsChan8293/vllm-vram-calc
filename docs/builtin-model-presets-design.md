# 内置模型预设设计方案（精简版）

**日期**: 2026-03-07  
**状态**: 修订版 - 仅内置 Model ID

---

## 概述

在 HuggingFace 输入框上方添加厂商分组的下拉选择器，用户选择后自动填充 model ID，再手动 fetch 获取参数。

### 核心变更

| 对比项 | 原方案 | 新方案 |
|--------|--------|--------|
| 内置数据 | 完整参数（weights, layers, heads...） | 仅 model ID |
| 数据来源 | 手动维护 | HuggingFace API |
| 代码量 | ~500 行 | ~80 行 |
| 维护成本 | 高（参数变更需更新） | 零 |

### 目标

1. **快速选择**: 用户可一键选择热门模型的 ID
2. **零维护**: 无需手动维护模型参数
3. **数据准确**: 从 HuggingFace 获取最新参数
4. **简洁实现**: 最小化代码改动

---

## 数据结构

```javascript
const BUILTIN_MODELS = {
  deepseek: {
    name: 'DeepSeek',
    models: [
      { id: 'deepseek-ai/DeepSeek-V3.2', name: 'DeepSeek-V3.2' },
      { id: 'deepseek-ai/DeepSeek-R1', name: 'DeepSeek-R1' }
    ]
  },
  qwen: {
    name: 'Qwen',
    models: [
      { id: 'Qwen/Qwen3-32B', name: 'Qwen3-32B' },
      { id: 'Qwen/Qwen3-14B', name: 'Qwen3-14B' }
    ]
  },
  glm: {
    name: 'GLM',
    models: [
      { id: 'zai-org/GLM-5', name: 'GLM-5' },
      { id: 'zai-org/GLM-4.7', name: 'GLM-4.7' }
    ]
  }
};
```

---

## UI 设计

### 布局

```
┌─────────────────────────────────────────┐
│ 快速选择模型                              │
│                                         │
│ DeepSeek    [▼ 选择模型...            ] │
│ Qwen        [▼ 选择模型...            ] │
│ GLM         [▼ 选择模型...            ] │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 从 HuggingFace 获取模型参数              │
│                                         │
│ Model ID: [________________] [Fetch]   │
└─────────────────────────────────────────┘
```

### HTML 结构

**位置**: 在 HuggingFace 输入框 `<div class="form-group">` 之前插入

```html
<!-- 内置模型快速选择 -->
<div class="form-group builtin-presets-group">
  <label>
    快速选择模型
    <span class="tooltip" data-tooltip="选择后自动填充 Model ID，点击 Fetch 获取参数">
      <i data-lucide="help-circle" style="width: 14px; height: 14px;"></i>
    </span>
  </label>
  
  <div class="vendor-select-row">
    <span class="vendor-label">DeepSeek</span>
    <select class="builtin-model-select" id="builtin-deepseek">
      <option value="">选择模型...</option>
      <option value="deepseek-ai/DeepSeek-V3.2">DeepSeek-V3.2</option>
      <option value="deepseek-ai/DeepSeek-R1">DeepSeek-R1</option>
    </select>
  </div>
  
  <div class="vendor-select-row">
    <span class="vendor-label">Qwen</span>
    <select class="builtin-model-select" id="builtin-qwen">
      <option value="">选择模型...</option>
      <option value="Qwen/Qwen3-32B">Qwen3-32B</option>
      <option value="Qwen/Qwen3-14B">Qwen3-14B</option>
    </select>
  </div>
  
  <div class="vendor-select-row">
    <span class="vendor-label">GLM</span>
    <select class="builtin-model-select" id="builtin-glm">
      <option value="">选择模型...</option>
      <option value="zai-org/GLM-5">GLM-5</option>
      <option value="zai-org/GLM-4.7">GLM-4.7</option>
    </select>
  </div>
</div>
```

### CSS 样式

```css
/* 内置模型快速选择 */
.builtin-presets-group {
  margin-bottom: 1.5rem;
}

.vendor-select-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.vendor-select-row:last-child {
  margin-bottom: 0;
}

.vendor-label {
  min-width: 80px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.builtin-model-select {
  flex: 1;
  padding: 0.5rem 0.75rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-primary);
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s;
}

.builtin-model-select:hover {
  border-color: var(--accent-purple);
}

.builtin-model-select:focus {
  outline: none;
  border-color: var(--accent-purple);
  box-shadow: 0 0 0 3px rgba(170, 102, 255, 0.2);
}

@media (max-width: 768px) {
  .vendor-select-row {
    flex-direction: column;
    align-items: stretch;
    gap: 0.5rem;
  }
  
  .vendor-label {
    min-width: auto;
  }
}
```

---

## JavaScript 实现

```javascript
// 初始化内置模型快速选择
function initBuiltinModelSelects() {
  document.querySelectorAll('.builtin-model-select').forEach(select => {
    select.addEventListener('change', () => {
      const modelId = select.value;
      if (modelId) {
        // 填充到 HuggingFace 输入框
        const hfInput = document.getElementById('huggingface-model');
        if (hfInput) {
          hfInput.value = modelId;
          // 聚焦到输入框，方便用户查看
          hfInput.focus();
        }
        // 不自动 fetch，用户手动点击
      }
    });
  });
}
```

**初始化位置**: 在 `lucide.createIcons()` 之后调用

```javascript
lucide.createIcons();
initBuiltinModelSelects();
```

---

## 交互流程

```
用户在下拉框选择 "DeepSeek-V3.2"
    ↓
填充 model ID 到 HuggingFace 输入框
    ↓
输入框获得焦点
    ↓
用户检查 ID 正确后点击 "Fetch" 按钮
    ↓
调用 HuggingFace API 获取参数（现有逻辑）
    ↓
填充表单字段并触发计算
```

---

## 与现有功能的关系

### HuggingFace Fetch

完全复用现有 fetch 逻辑，零改动：

```
内置模型选择 ──→ 填充 Model ID ──→ 用户点击 Fetch ──→ 现有 fetch 逻辑
```

### 优势

1. **无冲突**: 不改变任何现有功能
2. **可叠加**: 用户仍可手动输入任意 model ID
3. **零依赖**: 不依赖外部数据源的正确性

---

## 测试清单

- [ ] 选择 DeepSeek 模型 → ID 正确填充
- [ ] 选择 Qwen 模型 → ID 正确填充
- [ ] 选择 GLM 模型 → ID 正确填充
- [ ] 切换下拉框 → 输入框值更新
- [ ] 选择后聚焦到输入框
- [ ] 点击 Fetch → 正确获取参数
- [ ] 手动修改输入框 → fetch 仍正常工作
- [ ] 移动端布局正常

---

## 集成点

| 位置 | 变更类型 | 行数 |
|------|----------|------|
| `index.html` HTML | 插入 | ~30 行 |
| `index.html` CSS | 添加 | ~35 行 |
| `index.html` JS | 添加 | ~15 行 |
| `index.html` 初始化 | 添加 | 1 行 |

**总增量**: ~80 行

---

## 部署

- **环境**: 纯静态 HTML
- **依赖**: 无新增
- **托管**: GitHub Pages

---

## 审批

- [x] 设计已审核并批准
- [ ] 实现完成
- [ ] 测试通过