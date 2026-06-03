"""Base Page Object classes for Playwright and Selenium.

Generated automation scripts should subclass these to inherit
smart waits, retry logic, and consistent selector strategies.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page as PWPage
    from selenium.webdriver.remote.webdriver import WebDriver


# ---------------------------------------------------------------------------
# Playwright Base Page
# ---------------------------------------------------------------------------

class PlaywrightBasePage:
    """Base POM for Playwright-based tests.

    Features:
    - Auto-wait via Playwright's built-in actionability checks
    - No hard sleeps — uses ``expect`` + ``wait_for_selector``
    - Smart retry for flaky element interactions
    """

    BASE_URL: str = os.getenv("TARGET_BASE_URL", os.getenv("BASE_URL", "http://localhost:3000")).rstrip("/")

    # Default per-action / navigation timeouts (ms). Exposed as instance
    # attributes so page objects can reference ``self.timeout`` directly.
    timeout: int = 30_000
    navigation_timeout: int = 30_000

    def __init__(self, page: "PWPage", base_url: str | None = None) -> None:
        self.page = page
        if base_url:
            self.BASE_URL = base_url
        self.timeout = 30_000
        self.navigation_timeout = 30_000
        self.page.set_default_timeout(self.timeout)
        self.page.set_default_navigation_timeout(self.navigation_timeout)

    # -- Navigation ----------------------------------------------------------

    def navigate(self, path: str = "/") -> None:
        url = f"{self.BASE_URL}{path}"
        self.page.goto(url, wait_until="domcontentloaded")

    def navigate_sidebar(self, item_text: str, path: str | None = None) -> None:
        """Navigate to a section via the left sidebar.

        Tries clicking the sidebar entry whose visible text matches
        ``item_text``; if no such entry is found, falls back to navigating
        directly to ``path`` (reliable once authenticated). Generated page
        objects call this, so it must exist on the base class.
        """
        candidates = (
            f"a:has-text('{item_text}')",
            f"[role='tab']:has-text('{item_text}')",
            f"span:text-is('{item_text}')",
            f"li:has-text('{item_text}')",
            f"nav >> text={item_text}",
        )
        for sel in candidates:
            try:
                loc = self.page.locator(sel).first
                if loc.count() > 0:
                    loc.click(timeout=5_000)
                    self.page.wait_for_load_state("networkidle")
                    return
            except Exception:
                continue
        if path:
            self.navigate(path)
            try:
                self.page.wait_for_load_state("networkidle")
            except Exception:
                pass

    # -- Element helpers (no hard waits) ------------------------------------

    def fill_by_label(self, label: str, value: str) -> None:
        """Fill the input associated with a visible ``<label>`` text.

        Use this for fields identified by their label (e.g. "Email Address")
        instead of ``locator("label:has-text(...)")``, which targets the label
        element itself (not fillable).
        """
        self.page.get_by_label(label).fill(value)

    def click(self, selector: str) -> None:
        self.page.locator(selector).click()

    def fill(self, selector: str, value: str) -> None:
        locator = self.page.locator(selector)
        locator.fill(value)

    def get_text(self, selector: str) -> str:
        return self.page.locator(selector).inner_text()

    def is_visible(self, selector: str, timeout: int = 5_000) -> bool:
        try:
            self.page.locator(selector).wait_for(state="visible", timeout=timeout)
            return True
        except Exception:
            return False

    def wait_for(self, selector: str, state: str = "visible", timeout: int = 10_000) -> None:
        self.page.locator(selector).wait_for(state=state, timeout=timeout)

    def select_option(self, selector: str, value: str) -> None:
        self.page.locator(selector).select_option(value)

    def expect_element_to_be_visible(self, selector: str, timeout: int = 10_000) -> None:
        self.page.locator(selector).wait_for(state="visible", timeout=timeout)

    def click_element(self, selector: str) -> None:
        self.page.locator(selector).click()

    def fill_input(self, selector: str, value: str) -> None:
        locator = self.page.locator(selector)
        locator.fill(value)

    def screenshot(self, name: str = "screenshot") -> bytes:
        return self.page.screenshot(full_page=True)


# ---------------------------------------------------------------------------
# Selenium Base Page
# ---------------------------------------------------------------------------

class SeleniumBasePage:
    """Base POM for Selenium-based tests.

    Features:
    - Explicit waits via ``WebDriverWait`` (never ``time.sleep``)
    - Robust selector strategy (data-testid → aria → CSS)
    - Auto-scrolling before interactions
    """

    BASE_URL: str = os.getenv("TARGET_BASE_URL", os.getenv("BASE_URL", "http://localhost:3000")).rstrip("/")

    def __init__(self, driver: "WebDriver", base_url: str | None = None) -> None:
        self.driver = driver
        if base_url:
            self.BASE_URL = base_url
        self.driver.implicitly_wait(0)  # We use explicit waits only

    # -- Navigation ----------------------------------------------------------

    def navigate(self, path: str = "/") -> None:
        self.driver.get(f"{self.BASE_URL}{path}")

    # -- Element helpers (explicit waits only) -------------------------------

    def _wait_for_element(self, by: str, value: str, timeout: int = 10):
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )

    def _wait_clickable(self, by: str, value: str, timeout: int = 10):
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        return WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )

    def click(self, by: str, value: str) -> None:
        el = self._wait_clickable(by, value)
        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        el.click()

    def fill(self, by: str, value: str, text: str) -> None:
        el = self._wait_for_element(by, value)
        el.clear()
        el.send_keys(text)

    def get_text(self, by: str, value: str) -> str:
        return self._wait_for_element(by, value).text

    def is_visible(self, by: str, value: str, timeout: int = 5) -> bool:
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located((by, value))
            )
            return True
        except Exception:
            return False

    def expect_element_to_be_visible(self, by: str, value: str, timeout: int = 10) -> None:
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        WebDriverWait(self.driver, timeout).until(
            EC.visibility_of_element_located((by, value))
        )

    def click_element(self, by: str, value: str) -> None:
        self.click(by, value)

    def fill_input(self, by: str, value: str, text: str) -> None:
        self.fill(by, value, text)

    def screenshot(self, name: str = "screenshot") -> bool:
        return self.driver.save_screenshot(f"{name}.png")
