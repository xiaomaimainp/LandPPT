"""
Create default global master template
"""

import asyncio
import logging
from .database import AsyncSessionLocal
from .service import DatabaseService

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/js/all.min.js"></script>
    <style>
        body {
            width: 1280px;
            height: 720px;
            margin: 0;
            padding: 0;
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            font-family: 'Microsoft YaHei', 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
        }
        
        .slide-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            color: white;
            position: relative;
        }
        
        .slide-header {
            padding: 40px 60px 20px 60px;
            border-bottom: 2px solid rgba(96, 165, 250, 0.3);
        }
        
        .slide-title {
            font-size: clamp(2rem, 4vw, 3.5rem);
            font-weight: bold;
            color: #60a5fa;
            margin: 0;
            line-height: 1.2;
            max-height: 80px;
            overflow: hidden;
        }
        
        .slide-content {
            flex: 1;
            padding: 30px 60px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            max-height: 580px;
            overflow: hidden;
        }
        
        .content-main {
            font-size: clamp(1rem, 2.5vw, 1.4rem);
            line-height: 1.5;
            color: #e2e8f0;
        }
        
        .content-points {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .content-points li {
            margin-bottom: 15px;
            padding-left: 30px;
            position: relative;
        }
        
        .content-points li:before {
            content: "▶";
            position: absolute;
            left: 0;
            color: #60a5fa;
            font-size: 0.8em;
        }
        
        .slide-footer {
            position: absolute;
            bottom: 20px;
            right: 30px;
            font-size: 14px;
            color: #94a3b8;
            font-weight: 600;
        }
        
        .chart-container {
            max-height: 300px;
            margin: 20px 0;
        }
        
        .highlight-box {
            background: rgba(96, 165, 250, 0.1);
            border-left: 4px solid #60a5fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(96, 165, 250, 0.3);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #60a5fa;
            display: block;
        }
        
        .stat-label {
            font-size: 1rem;
            color: #cbd5e1;
            margin-top: 5px;
        }
        
        /* 响应式调整 */
        @media (max-width: 1280px) {
            body {
                width: 100vw;
                height: 56.25vw;
                max-height: 100vh;
            }
        }
    </style>
</head>
<body>
    <div class="slide-container">
        <div class="slide-header">
            <h1 class="slide-title">{{ main_heading }}</h1>
        </div>
        
        <div class="slide-content">
            <div class="content-main">
                {{ page_content }}
            </div>
        </div>
        
        <div class="slide-footer">
            {{ current_page_number }} / {{ total_page_count }}
        </div>
    </div>
</body>
</html>"""

async def create_default_global_template():
    """Create default global master template"""
    try:
        async with AsyncSessionLocal() as session:
            db_service = DatabaseService(session)
            
            # Check if default template already exists
            existing = await db_service.get_global_master_template_by_name("默认商务模板")
            if existing:
                logger.info("Default global master template already exists")
                return existing.id
            
            # Create default template
            template_data = {
                "template_name": "默认商务模板",
                "description": "现代简约的商务PPT模板，适用于各种商务演示场景。采用深色背景和蓝色主色调，支持多种内容类型展示。",
                "html_template": DEFAULT_TEMPLATE_HTML,
                "tags": ["默认", "商务", "现代", "简约", "深色"],
                "is_default": True,
                "is_active": True,
                "created_by": "system"
            }
            
            template = await db_service.create_global_master_template(template_data)
            logger.info(f"Created default global master template with ID: {template.id}")
            return template.id
            
    except Exception as e:
        logger.error(f"Error creating default global master template: {e}")
        raise

async def ensure_default_template_exists():
    """Ensure default template exists, create if not"""
    try:
        template_id = await create_default_global_template()
        logger.info(f"Default global master template ensured with ID: {template_id}")
        return template_id
    except Exception as e:
        logger.error(f"Failed to ensure default template exists: {e}")
        return None

if __name__ == "__main__":
    # Run the script to create default template
    asyncio.run(ensure_default_template_exists())
