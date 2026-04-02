# Spec 文档格式规范

<!-- DOC_SUMMARY_START -->
定义 SimpleLLMFunc 项目中所有 spec 文档的格式规范，包括文档头部、Section 格式、标记说明等内容。本规范用于确保 Agent 能够一致地读取、理解和修改 spec 文档。
<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->
## 文档目录 (Document Map)

- [文档头部格式](#文档头部格式)
  - [DOC_SUMMARY](#doc_summary)
  - [DOC_MAP](#doc_map)
  - [DOC_META（可选）](#doc_meta可选)
- [Section 格式](#section-格式)
  - [Section 说明](#section-说明)
- [标记说明](#标记说明)
- [Grep 查询示例](#grep-查询示例)
- [内容组织原则](#内容组织原则)
- [代码示例格式](#代码示例格式)
- [模块 Spec 文件存放位置](#模块-spec-文件存放位置)
- [文档更新规范](#文档更新规范)
<!-- DOC_MAP_END -->

<!-- DOC_META_START -->
## 文档更新规范

### 更新时机
- 添加新模块时，需要在 project-map.md 中添加模块信息
- 修改模块结构时，需要同步更新 project-map.md
- 添加新的最佳实践时，需要在 overall-spec.md 中添加相应章节

### 修改规范
- 保持文档结构一致性，遵循 meta.md 中的格式规范
- 更新 DOC_MAP 时，确保所有链接的锚点 ID 正确
- 更新后检查 lint 错误，修复格式问题

### 相关文件
- 本文档: spec/meta.md
- 相关规范: spec/meta.md
- 相关文档: spec/project-map.md, spec/overall-spec.md
<!-- DOC_META_END -->

## 文档头部格式 {#document-header}

每个 spec 文档必须以以下格式开头：

```markdown
# [文档标题]

<!-- DOC_SUMMARY_START -->
[文档的简要描述，用于快速了解文档内容]
<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->
## 文档目录 (Document Map)

- [Section 1](#section-1-id)
  - [Subsection 1.1](#subsection-1-1-id)
  - [Subsection 1.2](#subsection-1-2-id)
- [Section 2](#section-2-id)
  - [Subsection 2.1](#subsection-2-1-id)
<!-- DOC_MAP_END -->

<!-- DOC_META_START -->
## 文档更新规范

### 更新时机
- [该文档的更新时机说明]

### 修改规范
- [该文档的修改规范说明]

### 相关文件
- 本文档: [文档路径]
- 相关规范: spec/meta.md
- 相关文档: [其他相关文档路径]
<!-- DOC_META_END -->
```

### 头部说明 {#header-intro}

#### DOC_SUMMARY {#doc_summary}

文档的简要描述，1-3 句话，方便 Agent 快速判断文档是否相关。

- 必须包含在 `<!-- DOC_SUMMARY_START -->` 和 `<!-- DOC_SUMMARY_END -->` 之间
- 应简洁明了，概括文档的核心内容

#### DOC_MAP {#doc_map}

文档的完整目录结构，使用 Markdown 链接格式，方便 Agent 通过 grep 提取并导航。

- 必须包含在 `<!-- DOC_MAP_START -->` 和 `<!-- DOC_MAP_END -->` 之间
- 使用嵌套列表展示文档的层次结构
- 所有链接的锚点 ID 必须使用英文，格式为 `{#id}`

#### DOC_META（可选） {#doc-meta-optional}

文档元信息，包含文档的更新和修改规范，仅在需要 Agent 修改文档时使用。

**格式要求**：

```markdown
<!-- DOC_META_START -->
## 文档更新规范

### 更新时机
- [该文档的更新时机说明]

### 修改规范
- [该文档的修改规范说明]

### 相关文件
- 本文档: [文档路径]
- 相关规范: spec/meta.md
- 相关文档: [其他相关文档路径]
<!-- DOC_META_END -->
```

**使用说明**：

- DOC_META 部分用 `<!-- DOC_META_START -->` 和 `<!-- DOC_META_END -->` 包裹
- 位于 DOC_MAP 之后，正文内容之前
- 大部分情况下 Agent 不需要读取这部分内容
- 仅在需要 Agent 帮助修改 Spec 文档内容时（如添加新模块、更新规范等）才会用到
- 可以通过 `grep -A 50 "DOC_META_START"` 提取文档元信息

**内容要求**：

- **更新时机**: 说明在什么情况下需要更新该文档
- **修改规范**: 说明修改文档时应遵循的规范和注意事项
- **相关文件**: 列出与该文档相关的其他文档路径

## Section 格式 {#section-format}

每个主要 Section 必须遵循以下格式：

```markdown
## [Section 标题] {#section-id}

<!-- SECTION_SUMMARY_START -->
[该 Section 的简要描述，1-2 句话]
<!-- SECTION_SUMMARY_END -->

<!-- SECTION_TOC_START -->
### 本节目录

- [Subsection 1](#subsection-1-id)
- [Subsection 2](#subsection-2-id)
<!-- SECTION_TOC_END -->

### [Subsection 标题] {#subsection-id}

[具体内容...]
```

### Section 说明 {#section-intro}

- **SECTION_SUMMARY**: 该 Section 的简要描述，帮助 Agent 判断是否需要深入阅读
- **SECTION_TOC**: 该 Section 的子目录，方便快速定位
- **锚点 ID**: 使用 `{#id}` 格式，确保链接可跳转

## 标记说明 {#marker-description}

所有标记使用 HTML 注释格式，方便 grep 提取：

- `<!-- DOC_SUMMARY_START -->` / `<!-- DOC_SUMMARY_END -->`: 文档摘要
- `<!-- DOC_MAP_START -->` / `<!-- DOC_MAP_END -->`: 文档目录
- `<!-- DOC_META_START -->` / `<!-- DOC_META_END -->`: 文档元信息（更新规范）
- `<!-- SECTION_SUMMARY_START -->` / `<!-- SECTION_SUMMARY_END -->`: Section 摘要
- `<!-- SECTION_TOC_START -->` / `<!-- SECTION_TOC_END -->`: Section 目录

## Grep 查询示例 {#grep-examples}

### 提取文档摘要

```bash
grep -A 1 "DOC_SUMMARY_START" spec/*.md
```

### 提取文档目录

```bash
grep -A 20 "DOC_MAP_START" spec/*.md
```

### 提取特定 Section 摘要

```bash
grep -A 1 "SECTION_SUMMARY_START" spec/overall-spec.md | grep -A 1 "模块化架构"
```

## 内容组织原则 {#content-principles}

1. **渐进式展开**: 从概述到细节，从通用到具体
2. **层次清晰**: 使用明确的标题层级（##, ###, ####）
3. **可搜索性**: 关键术语和概念使用标准命名，便于搜索
4. **可引用性**: 每个重要概念都有明确的章节和锚点

## 代码示例格式 {#code-example-format}

代码示例应包含：

- 完整的上下文（imports, decorators 等）
- 注释说明关键点
- 错误示例（如果适用）

```typescript
// ✅ 正确示例
@Module({
  imports: [ConfigModule],
  providers: [MyService],
})
export class MyModule {}

// ❌ 错误示例
@Module({
  providers: [MyService], // 缺少必要的 imports
})
```

## 模块 Spec 文件存放位置 {#spec-file-location}

每个模块的详细 spec 文档应存放在对应模块目录下的 `spec/` 文件夹中：

### 业务模块

- **路径**: `src/modules/{module-name}/spec/`
- **示例**:
  - `src/modules/apps/spec/apps-spec.md`
  - `src/modules/user/spec/user-spec.md`
  - `src/modules/proxy/spec/proxy-spec.md`

### 核心模块

- **路径**: `src/core/{module-name}/spec/`
- **示例**:
  - `src/core/auth/spec/auth-spec.md`
  - `src/core/oss/spec/oss-spec.md`
  - `src/core/redis/spec/redis-spec.md`

### 公共代码

- **路径**: `src/common/{category}/spec/`
- **示例**:
  - `src/common/base/spec/result-spec.md`
  - `src/common/config/spec/config-spec.md`

### 规范说明

1. **目录结构**: 每个模块目录下创建 `spec/` 子目录存放该模块的 spec 文档
2. **文件命名**: 使用 `{module-name}-spec.md` 或 `{category}-spec.md` 格式
3. **文档格式**: 遵循本 meta.md 中定义的格式规范
4. **查找方式**: Agent 可以通过 `project-map.md` 找到模块路径，然后在对应路径下的 `spec/` 目录查找详细规范

## 文档更新规范 {#update-rules}

### 更新时机

以下情况需要更新 Spec 文档：

1. **添加新模块**:
   - 在 `project-map.md` 中添加新模块的路径和作用说明
   - 在新模块目录下创建 `spec/` 文件夹和对应的 spec 文档

2. **修改模块结构**:
   - 更新 `project-map.md` 中对应模块的描述
   - 同步更新模块目录下的 spec 文档

3. **添加新的最佳实践**:
   - 在 `overall-spec.md` 中添加相应章节
   - 更新 `overall-spec.md` 的 DOC_MAP

4. **修改项目架构**:
   - 更新 `project-map.md` 中的项目结构说明
   - 更新相关模块的 spec 文档

### 修改规范

1. **保持格式一致性**:
   - 遵循 `meta.md` 中定义的格式规范
   - 确保所有标记（DOC_SUMMARY, DOC_MAP, SECTION_SUMMARY 等）格式正确

2. **更新目录结构**:
   - 修改内容后，同步更新 DOC_MAP 和 SECTION_TOC
   - 确保所有链接的锚点 ID 正确且可跳转

3. **检查完整性**:
   - 更新后检查 lint 错误
   - 确保所有章节都有对应的摘要和目录
   - 验证代码示例格式正确

4. **文档关联性**:
   - 确保 `project-map.md` 中的模块路径与实际目录结构一致
   - 确保模块 spec 文档与 `overall-spec.md` 中的通用规范一致
