#!/usr/bin/env python3
"""
Simple test script to verify browser configuration works in Cloud Run environment.
"""
import asyncio
import logging
from crawl4ai import AsyncWebCrawler, BrowserConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_browser_config():
    """Test if the browser configuration works."""
    playwright_launch_args = [
        # --- Core Security & Sandboxing ---
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--single-process',
        '--no-zygote',
        
        # --- GPU & Rendering (optimized for headless) ---
        '--disable-gpu',
        '--disable-gpu-sandbox',
        '--disable-gpu-compositing',
        '--disable-software-rasterizer',
        '--disable-webgl',
        '--disable-webgl2',
        '--use-gl=swiftshader',

        # --- Memory & Performance ---
        '--memory-pressure-off',
        '--disk-cache-size=0',
        '--media-cache-size=0',
        
        # --- Network & Services ---
        '--disable-background-networking',
        '--disable-extensions',
        '--disable-dbus',
        '--disable-logging',
        '--log-level=3',
        '--headless',
    ]
    
    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        extra_args=playwright_launch_args,
        verbose=False,
        text_mode=True,
        light_mode=True,
    )
    
    try:
        logger.info("Testing browser initialization...")
        async with AsyncWebCrawler(verbose=False, config=browser_config) as crawler:
            logger.info("✅ Browser initialized successfully!")
            
            # Test basic navigation
            logger.info("Testing basic navigation...")
            result = await crawler.arun("https://www.google.com")
            
            if result.success:
                logger.info("✅ Navigation test successful!")
                logger.info(f"Page title length: {len(result.markdown) if result.markdown else 0}")
                return True
            else:
                logger.error(f"❌ Navigation failed: {result.error_message}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Browser test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_browser_config())
    if result:
        print("🎉 All browser tests passed!")
        exit(0)
    else:
        print("💥 Browser tests failed!")
        exit(1) 