以下是使用Markdown格式编写的上述Python代码功能和使用文档：

---

## 使用文档

这段Python代码提供了两个函数，用于在终端中打印彩色的文本框，以增强文本信息的视觉表现。以下是每个函数的详细说明和使用示例。

### 1. `print_colored_box_line` 函数

#### 功能描述

`print_colored_box_line` 函数用于在终端打印一个含有标题和消息的彩色文本框。这个文本框具有上下边框、居中的标题和消息，用户可以自定义文本和框的颜色及属性（例如加粗）。此函数适用于需要强调显示重要信息或区分文本段落的场景。

#### 参数说明

- `title`: 文本框中心的标题文本。
- `message`: 文本框中的消息文本。
- `attrs`: 文本属性列表（例如 `['bold']` 表示加粗文本）。
- `text_color`: 文本的颜色。
- `box_color`: 文本框的颜色。
- `box_width`: 文本框的宽度，默认值为80字符。

#### 使用示例

```python
from printk import print_colored_box, print_colored_box_line

print_colored_box_line("警告", "请立即检查系统！", attrs=['bold'], text_color='red', box_color='yellow')
```

此示例将在终端打印一个黄色背景框，其中包含红色加粗的“警告”标题和“请立即检查系统！”的消息。

### 2. `print_colored_box` 函数

#### 功能描述

`print_colored_box` 函数用于在终端打印一个彩色边框的文本框，可以选择为文本添加背景颜色。此函数适合用于突出显示单条信息，通过颜色和边框的变化来吸引用户的注意。

#### 参数说明

- `text`: 文本框中显示的文本。
- `pad_len`: 用户指定的输出宽度。
- `text_color`: 文本的颜色。
- `box_color`: 边框的颜色。
- `background_color`: 文本的背景颜色，仅在 `text_background` 为 True 时适用。
- `text_background`: 布尔值，指定是否为文本添加背景颜色。



#### 使用示例

```python
print_colored_box("操作成功", text_background=True, text_color='green', box_color='green', background_color='on_white')
```

此示例将在终端打印一个绿色边框和文本的文本框，如果支持的话，文本背景为白色。

---

### 概述

通过使用这两个函数，开发者可以在他们的Python应用程序中以视觉上引人注目的方式显示信息，如日志消息、警告或成功提示。自定义文本、背景和边框颜色以及文本属性使得集成这些彩色文本框变得简单，从而提升用户的交互体验。