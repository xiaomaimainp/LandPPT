"""
提示模板 - 定义所有LLM交互的提示模板
"""

from langchain_core.prompts import ChatPromptTemplate


class PromptTemplates:
    """提示模板集合"""
    
    @staticmethod
    def get_structure_analysis_prompt() -> ChatPromptTemplate:
        """文档结构分析提示"""
        return ChatPromptTemplate([
            ("human", """
            请分析以下文档片段，提取文档结构信息。

            文档内容：
            {content}

            请识别并返回JSON格式的结果，包含以下字段：
            1. title: 文档标题（如果存在）
            2. type: 文档类型（如：学术论文、技术报告、商业文档、教程等）
            3. sections: 主要章节结构列表
            4. key_concepts: 关键概念和主题列表
            5. language: 文档原始语言（仅用于参考，不影响生成语言）
            6. complexity: 复杂度等级（简单/中等/复杂）
            7. key_data: 文档中的关键数据和统计信息
            8. main_conclusions: 主要结论和观点

            **重要说明：**
            - 提取具体的数据、数字、百分比等关键信息
            - 识别文档的核心结论和重要观点

            示例输出格式：
            {{
                "title": "深度学习在自然语言处理中的应用",
                "type": "学术论文",
                "sections": ["摘要", "引言", "相关工作", "方法论", "实验结果", "结论"],
                "key_concepts": ["深度学习", "自然语言处理", "神经网络", "机器学习"],
                "language": "中文",
                "complexity": "复杂",
                "key_data": ["准确率提升15%", "训练时间减少30%", "数据集包含10万条样本"],
                "main_conclusions": ["深度学习显著提升了NLP任务性能", "Transformer架构是当前最优选择"]
            }}

            请确保返回有效的JSON格式。
            """)
        ])
    
    @staticmethod
    def get_initial_outline_prompt() -> ChatPromptTemplate:
        """初始PPT框架生成提示"""
        return ChatPromptTemplate([
            ("human", """
            基于以下文档结构信息和内容，生成PPT的初始框架。

            文档结构：
            {structure}

            文档内容：
            {content}

            页数要求：
            {slides_range}

            目标语言：
            {target_language}

            **要求：**
            1. **语言一致性**：必须严格按照目标语言({target_language})生成所有内容
            2. **内容丰富性与创意性**：每个幻灯片的content_points要充分发挥创意和设计能力：
               - 包含具体的关键数据、数字、百分比、统计信息
               - 融入重要结论、核心观点、深度洞察
               - 添加生动的案例、实例、对比分析
               - 引用相关的内容片段、权威观点
               - 使用富有表现力的描述，增强内容吸引力
               - 结合行业趋势、前沿技术、创新思维
               - 提供多角度分析，展现内容的深度和广度
            3. **图表样式多样化**：在适当的内容页中添加chart_config字段，支持丰富的Chart.js、ECharts、D3.js图表样式：
               - 数据对比：柱状图(bar)、水平柱状图(horizontalBar)、堆叠柱状图、分组柱状图
               - 趋势分析：折线图(line)、面积图(area)、阶梯图、多轴图表
               - 比例分布：饼图(pie)、环形图(doughnut)、极地图(polarArea)、雷达图(radar)
               - 散点分析：散点图(scatter)、气泡图(bubble)
               - 混合图表：组合柱状图和折线图、多类型数据展示
               - 创新样式：渐变色彩、动态效果、交互元素

            请生成一个完整的PPT大纲，包含以下要求：
            1. 生成合适的PPT标题（使用目标语言：{target_language}）
            2. 规划幻灯片总数（严格控制在{slides_range}范围内）
            3. 创建详细的幻灯片内容，包括：
               - 标题页
               - 目录/议程页
               - 主要内容页（根据页数要求调整数量）
               - 结论页

            返回JSON格式，包含以下字段：
            - title: PPT标题（使用目标语言：{target_language}）
            - total_pages: 预计总页数
            - page_count_mode: "estimated"
            - slides: 幻灯片列表，每个幻灯片包含：
              - page_number: 页码
              - title: 幻灯片标题（使用目标语言：{target_language}）
              - content_points: 内容要点列表（包含数据、结论、片段）
              - slide_type: 类型（"title"/"content"/"conclusion"）
              - description: 幻灯片描述（使用目标语言：{target_language}）
              - chart_config: 可选，图表配置（当内容适合可视化时）

            示例输出格式：
            {{
                "title": "深度学习在自然语言处理中的应用研究",
                "total_pages": 15,
                "page_count_mode": "estimated",
                "slides": [
                    {{
                        "page_number": 1,
                        "title": "深度学习在自然语言处理中的应用研究",
                        "content_points": ["研究主题：基于Transformer的文本分析", "演示者：AI研究团队", "日期：2024年度报告"],
                        "slide_type": "title",
                        "description": "PPT标题页，介绍研究主题和基本信息"
                    }},
                    {{
                        "page_number": 2,
                        "title": "研究概览与创新突破",
                        "content_points": ["研究背景：NLP领域年增长率达35%，市场需求激增", "核心创新：基于Transformer的多模态融合架构", "突破性成果：准确率提升15%，达到业界领先水平", "应用价值：覆盖文本分类、情感分析等8大场景", "技术优势：支持100+语言，实现真正的全球化部署", "产业影响：预计为企业节省40%的人工成本"],
                        "slide_type": "content",
                        "description": "展示研究的创新性和突破性成果，突出技术价值和产业影响"
                    }},
                    {{
                        "page_number": 3,
                        "title": "性能革命性提升对比",
                        "content_points": ["传统方法局限：准确率仅78%，处理速度慢", "深度学习突破：准确率飙升至93%，行业新标杆", "效率革命：处理速度提升2.5倍，实时响应能力", "资源优化：内存使用减少40%，成本大幅降低", "稳定性提升：错误率下降60%，可靠性显著增强"],
                        "slide_type": "content",
                        "description": "通过对比展示技术革命性的性能提升和优势",
                        "chart_config": {{
                            "type": "bar",
                            "data": {{
                                "labels": ["传统方法", "深度学习方法"],
                                "datasets": [{{
                                    "label": "准确率(%)",
                                    "data": [78, 93],
                                    "backgroundColor": ["rgba(255, 99, 132, 0.8)", "rgba(54, 162, 235, 0.8)"],
                                    "borderColor": ["rgba(255, 99, 132, 1)", "rgba(54, 162, 235, 1)"],
                                    "borderWidth": 2
                                }}, {{
                                    "label": "处理速度(倍数)",
                                    "data": [1, 2.5],
                                    "backgroundColor": ["rgba(255, 206, 86, 0.8)", "rgba(75, 192, 192, 0.8)"],
                                    "borderColor": ["rgba(255, 206, 86, 1)", "rgba(75, 192, 192, 1)"],
                                    "borderWidth": 2
                                }}]
                            }},
                            "options": {{
                                "responsive": true,
                                "plugins": {{
                                    "legend": {{
                                        "position": "top"
                                    }},
                                    "title": {{
                                        "display": true,
                                        "text": "技术性能对比分析"
                                    }}
                                }},
                                "scales": {{
                                    "y": {{
                                        "beginAtZero": true
                                    }}
                                }}
                            }}
                        }}
                    }}
                ]
            }}

            **重要提醒：**
            - 所有文本内容必须使用目标语言：{target_language}
            - 内容要点要具体详实，充满创意和洞察力，包含数据、结论、案例
            - 充分发挥创意能力，生成富有表现力和吸引力的内容
            - 图表配置要多样化，使用丰富的样式和配色方案
            - 严格控制每页内容量，确保视觉效果完美

            请确保返回有效的JSON格式。
            """)
        ])
    
    @staticmethod
    def get_refine_outline_prompt() -> ChatPromptTemplate:
        """内容细化提示"""
        return ChatPromptTemplate([
            ("human", """
            基于已有的PPT大纲和新的文档内容，细化和扩展PPT结构。

            现有PPT结构：
            {existing_outline}

            新增内容：
            {new_content}

            累积上下文：
            {context}

            页数要求：
            {slides_range}

            目标语言：
            {target_language}

            **要求：**
            1. **语言一致性**：必须严格使用目标语言({target_language})，所有新增和修改内容都要使用此语言
            2. **内容深度挖掘与创意表达**：细化内容时要充分发挥AI的创意和分析能力：
               - 深度挖掘新内容中的关键数据、统计信息、趋势洞察
               - 提炼重要结论、核心观点、创新思维、前瞻性分析
               - 运用创意性的表达方式，增强内容的吸引力和说服力
               - 结合多维度分析，展现内容的深度、广度和高度
            3. **图表样式创新与多样化**：为适合的内容添加丰富的chart_config配置：
               - 数据对比：多样化柱状图、堆叠图、分组对比、渐变效果
               - 趋势分析：动态折线图、面积图、多轴展示、预测曲线
               - 比例分布：创意饼图、环形图、极地图、层次结构图
               - 性能展示：雷达图、散点图、气泡图、热力图
               - 混合图表：组合多种图表类型，创造视觉冲击力
               - 样式创新：使用渐变色彩、动画效果、交互元素

            请执行以下任务：
            1. 分析新增内容的主要信息点和关键数据
            2. 更新和完善现有幻灯片内容，增加具体数据和结论
            3. 根据新内容添加必要的幻灯片，包含丰富的信息
            4. 确保内容逻辑连贯性和流畅性
            5. 平衡各幻灯片间的内容分布，避免过度拥挤
            6. 为数据密集的内容添加图表配置

            返回完整的JSON格式PPT大纲，包含：
            - title: 更新后的PPT标题（使用目标语言：{target_language}）
            - total_pages: 更新后的总页数
            - page_count_mode: "refining"
            - slides: 完整的幻灯片列表（包含丰富内容和图表配置）

            注意事项：
            - 保持现有幻灯片的核心结构，但使用目标语言({target_language})
            - 新增的内容要与现有内容有机结合，形成完整的知识体系
            - 确保每张幻灯片内容量适中，但内容要充实有深度
            - 维护逻辑顺序和演示流程，增强内容的连贯性和说服力
            - 充分发挥创意能力，生成富有洞察力和表现力的内容
            - 图表配置要多样化创新，使用丰富的样式和视觉效果

            请确保返回有效的JSON格式。
            """)
        ])
    
    @staticmethod
    def get_finalize_outline_prompt() -> ChatPromptTemplate:
        """最终优化提示"""
        return ChatPromptTemplate([
            ("human", """
            对PPT大纲进行最终优化和完善。

            当前大纲：
            {outline}

            页数要求：
            {slides_range}

            目标语言：
            {target_language}

            **最终优化严格要求：**
            1. **语言一致性**：确保所有内容使用目标语言({target_language})
            2. **内容创意性与深度性**：每个要点要充分体现AI的创意和专业能力：
               - 融入关键数据、精确数字、重要百分比、统计洞察
               - 提供明确的结论、深度观点、创新思维、前瞻性分析
               - 包含生动案例、成功实例、对比研究、行业标杆
               - 运用富有表现力的语言，增强内容吸引力和说服力
               - 避免空泛的概念性描述，确保每个要点都有实质内容
               - 结合行业趋势、技术前沿、市场动态，展现专业深度
            3. **图表样式革新与完善**：为所有适合的数据内容添加创新的chart_config：
               - 确保图表类型与数据完美匹配，选择最佳展示方式
               - 提供完整的数据配置，包含丰富的样式设置
               - 使用渐变色彩、动态效果、交互元素增强视觉冲击力
               - 创新图表组合，如混合图表、多轴显示、层次结构
               - 优化图表布局和配色方案，确保专业美观
            4. **布局严格控制**：绝对避免滚动条出现：
               - 每页内容点：严格控制在3-6个
               - 每个要点：不超过50字符
               - 内容分布均匀，避免单页过载

            请执行以下优化任务：
            1. 确保幻灯片类型分布合理：
               - title: 标题页（1页）
               - content: 内容页（主体部分）
               - conclusion: 结论页（1-2页）

            2. 优化内容要点的创意表达：
               - 使用简洁明了但信息丰富、富有表现力的语言
               - 包含具体数据、深度结论、创新洞察
               - 适合演示和口头表达，具有强烈的视觉冲击力
               - 运用对比、类比、数据支撑等手法增强说服力

            3. 调整幻灯片顺序和逻辑流程：
               - 确保逻辑清晰
               - 内容递进合理
               - 重点突出，数据支撑

            4. 确保每张幻灯片内容量适中但信息丰富

            5. 添加必要的过渡和总结幻灯片，增强逻辑连贯

            6. 为每张幻灯片分配正确的页码

            7. 完善图表配置，确保数据可视化效果卓越：
               - 使用多样化的图表类型和创新样式
               - 优化配色方案和视觉效果
               - 增加交互元素和动态效果

            输出最终的JSON格式PPT大纲，包含：
            - title: 最终PPT标题
            - total_pages: 最终总页数
            - page_count_mode: "final"
            - slides: 优化后的完整幻灯片列表（包含丰富内容和图表）

            质量要求：
            - 内容专业且易懂，信息丰富，充满创意和洞察力
            - 结构清晰有逻辑，数据支撑，具有强烈的说服力
            - 适合演示展示，视觉效果卓越，具有专业水准
            - 充分发挥创意和设计能力，生成高质量内容
            - 严格遵守页数要求：{slides_range}
            - 严格使用目标语言：{target_language}

            请确保返回有效的JSON格式。
            """)
        ])
    
    @staticmethod
    def get_custom_prompt(template: str) -> ChatPromptTemplate:
        """自定义提示模板"""
        return ChatPromptTemplate([("human", template)])
    
    @staticmethod
    def get_error_recovery_prompt() -> ChatPromptTemplate:
        """错误恢复提示"""
        return ChatPromptTemplate([
            ("human", """
            之前的处理出现了错误，请基于以下信息生成一个基础的PPT大纲：

            文档内容摘要：
            {content_summary}

            错误信息：
            {error_info}

            目标语言：
            {target_language}

            **错误恢复要求：**
            1. **语言要求**：必须使用目标语言({target_language})生成所有内容
            2. **内容充实与创意性**：即使是基础大纲，也要充分发挥AI的创意能力：
               - 从摘要中深度提取关键数据、统计信息、重要洞察
               - 包含主要结论、核心观点、创新思维、价值主张
               - 添加具体的内容片段、生动案例、对比分析
               - 运用富有表现力的语言，增强内容吸引力
            3. **布局控制**：严格控制内容量，避免显示问题
            4. **图表创新支持**：为适合的内容添加多样化的图表配置：
               - 使用不同类型的图表展示数据
               - 优化配色和样式设计
               - 增强视觉表现力

            请生成一个简单但完整的PPT大纲，包含：
            1. 标题页（包含主题和基本信息）
            2. 目录页（包含主要章节）
            3. 3-5个主要内容页（包含数据和结论）
            4. 结论页（包含总结和要点）

            每个幻灯片要求：
            - 内容要点：3-6个，充满创意和专业洞察
            - 包含具体信息、数据支撑、深度分析，避免空泛描述
            - 适当添加多样化的图表配置，增强视觉效果
            - 严格控制内容长度，确保信息丰富但布局完美
            - 运用富有表现力的语言，体现AI的创意能力

            返回JSON格式，确保结构完整且有效，包含丰富的创意内容信息。
            """)
        ])
    
    @classmethod
    def get_all_prompts(cls) -> dict:
        """获取所有提示模板"""
        return {
            "structure_analysis": cls.get_structure_analysis_prompt(),
            "initial_outline": cls.get_initial_outline_prompt(),
            "refine_outline": cls.get_refine_outline_prompt(),
            "finalize_outline": cls.get_finalize_outline_prompt(),
            "error_recovery": cls.get_error_recovery_prompt(),
        }
