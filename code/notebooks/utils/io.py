def panel_to_png(fig, png_path="figures/fig1.png", width=850, header_height=90, timeout=40):
    import os, time, shutil
    from pathlib import Path
    import panel as pn
    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        import holoviews as hv
        hv.Store.renderers["bokeh"].webgl = False
    except Exception:
        pass

    # Export-only layout: keep top HTML pane from collapsing
    export_fig = fig
    if isinstance(fig, pn.Column) and len(fig) >= 2:
        first = fig[0]
        if isinstance(first, pn.pane.HTML):
            header = pn.pane.HTML(
                first.object, width=width, height=header_height,
                sizing_mode="fixed", margin=(0, 0, 10, 0)
            )
            export_fig = pn.Column(header, *list(fig[1:]), width=width, sizing_mode="fixed")

    png_path = Path(png_path).resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_tmp = png_path.with_suffix(".tmp_export.html")
    export_fig.save(str(html_tmp), resources="inline")

    firefox = shutil.which("firefox")
    gecko = shutil.which("geckodriver")
    if not firefox or not gecko:
        raise RuntimeError("firefox/geckodriver not found on PATH")

    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)

    opts = webdriver.FirefoxOptions()
    opts.binary_location = firefox
    opts.add_argument("-headless")
    opts.set_preference("network.proxy.type", 0)

    driver = webdriver.Firefox(service=Service(executable_path=gecko), options=opts)
    try:
        driver.get(f"file://{html_tmp}")
        wait = WebDriverWait(driver, timeout)
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        wait.until(lambda d: d.execute_script("return !!document.querySelector('[data-root-id]')"))

        # Critical: unhide hidden text containers inside shadow DOM
        driver.execute_script("""
        function walk(node){
          const kids = (node instanceof Document || node instanceof ShadowRoot || node instanceof Element) ? node.children : [];
          for (const el of kids){
            const cs = getComputedStyle(el);
            if (cs.visibility === 'hidden' && (el.textContent || '').trim().length) {
              el.style.visibility = 'visible';
            }
            if (el.shadowRoot) walk(el.shadowRoot);
            walk(el);
          }
        }
        walk(document);
        window.dispatchEvent(new Event('resize'));
        """)

        time.sleep(1.5)
        w = driver.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth, 1000)")
        h = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, 800)")
        driver.set_window_size(int(w + 40), int(h + 120))
        time.sleep(0.5)

        driver.save_screenshot(str(png_path))
    finally:
        driver.quit()
        html_tmp.unlink(missing_ok=True)

    return str(png_path)
