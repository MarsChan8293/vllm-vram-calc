# 内置模型预设快速选择 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 添加厂商分组的下拉选择器，用户选择后自动填充 HuggingFace Model ID

**Architecture:** 在 HuggingFace 输入框上方添加三个下拉选择器（DeepSeek/Qwen/GLM），选择后填充 model ID 到输入框，用户手动点击 fetch 获取参数

**Tech Stack:** Vanilla HTML/CSS/JavaScript, 无新增依赖

---

## Task 1: 添加 CSS 样式

**Files:**
- Modify: `index.html` (在 `<style>` 标签内)

**Step 1: 定位 CSS 插入位置**

找到 HuggingFace 相关样式附近，在合适位置添加新样式。

**Step 2: 添加内置模型选择器样式**

在 `<style>` 标签内添加以下 CSS：

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

**Step 3: 验证样式**

在浏览器中打开 `index.html`，确认样式无语法错误（可在开发者工具 Console 检查）。

**Step 4: 提交**

```bash
git add index.html
git commit -m "style: add builtin model select CSS styles"
```

---

## Task 2: 添加 HTML 结构

**Files:**
- Modify: `index.html` (在 HuggingFace 输入框之前)

**Step 1: 定位 HTML 插入位置**

找到 HuggingFace 输入框的 `<div class="form-group">` 标签，在其**之前**插入新的 HTML。

**Step 2: 添加内置模型选择器 HTML**

插入以下 HTML：

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

**Step 3: 验证 HTML 结构**

在浏览器中刷新页面，确认三个下拉选择器正确显示在 HuggingFace 输入框上方。

**Step 4: 提交**

```bash
git add index.html
git commit -m "feat: add builtin model select UI"
```

---

## Task 3: 添加 JavaScript 逻辑

**Files:**
- Modify: `index.html` (在 `<script>` 标签内)

**Step 1: 定位 JS 插入位置**

找到 `lucide.createIcons()` 调用附近，在初始化代码区域添加新函数。

**Step 2: 添加初始化函数**

在 `<script>` 标签内添加以下 JavaScript：

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
          // 聚焦到输入框
          hfInput.focus();
        }
      }
    });
  });
}
```

**Step 3: 添加初始化调用**

找到 `lucide.createIcons();` 行，在其后添加：

```javascript
// 初始化内置模型选择器
initBuiltinModelSelects();
```

**Step 4: 验证 JS 功能**

在浏览器中：
1. 打开开发者工具 Console
2. 选择任意下拉框中的模型
3. 确认 HuggingFace 输入框被填充正确的 model ID
4. 确认输入框获得焦点

**Step 5: 提交**

```bash
git add index.html
git commit -m "feat: implement builtin model select logic"
```

---

## Task 4: 完整功能测试

**Files:**
- Modify: `index.html`

**Step 1: 测试 DeepSeek 模型选择**

1. 选择 DeepSeek 下拉框中的 "DeepSeek-V3.2"
2. 确认输入框填充为 `deepseek-ai/DeepSeek-V3.2`
3. 点击 Fetch 按钮
4. 确认参数正确填充

**Step 2: 测试 Qwen 模型选择**

1. 选择 Qwen 下拉框中的 "Qwen3-32B"
2. 确认输入框更新为 `Qwen/Qwen3-32B`
3. 点击 Fetch
4. 确认参数正确填充

**Step 3: 测试 GLM 模型选择**

1. 选择 GLM 下拉框中的 "GLM-5"
2. 确认输入框更新为 `zai-org/GLM-5`
3. 点击 Fetch
4. 确认参数正确填充

**Step 4: 测试边界情况**

1. 手动修改输入框内容
2. 确认可以正常 Fetch 其他模型
3. 在移动端尺寸下测试布局是否正常

**Step 5: 最终提交**

```bash
git add index.html
git commit -m "feat: complete builtin model presets feature"
```

---

## Summary

**Total Changes:**
- CSS: ~35 lines
- HTML: ~30 lines  
- JS: ~15 lines
- **Total: ~80 lines**

**Files Modified:**
- `index.html` (only file)

**Testing:**
- 手动测试所有下拉选择
- 手动测试 Fetch 功能
- 响应式布局测试