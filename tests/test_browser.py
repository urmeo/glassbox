"""Optional browser contrast checks."""

import json
import os
import subprocess
import unittest

from glassbox import interfaces, render, schema


NODE = os.environ.get("GLASSBOX_BROWSER_NODE")
PUPPETEER = os.environ.get("GLASSBOX_BROWSER_PUPPETEER")
SCRIPT = r"""
const puppeteer = require(process.argv[1]);
const fs = require('fs');
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
function luminance(color) {
  const channels = color.match(/[\d.]+/g).slice(0, 3).map(value => {
    const scaled = Number(value) / 255;
    return scaled <= 0.04045 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4;
  });
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}
(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.argv[2], headless: true,
    args: ['--disable-background-networking', '--disable-component-update',
      '--disable-sync', '--no-first-run', '--no-default-browser-check',
      '--password-store=basic', '--use-mock-keychain']
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({width: 820, height: 620});
    let cases = 0;
    for (const fixture of fixtures) {
      for (const theme of ['light', 'dark']) {
        await page.emulateMediaFeatures([{name: 'prefers-color-scheme', value: theme}]);
        await page.setContent(fixture.html, {waitUntil: 'load'});
        const actual = await page.evaluate(() => {
          const plain = document.querySelector('pre.plain');
          return {text: plain.textContent, color: getComputedStyle(plain).color,
            background: getComputedStyle(document.body).backgroundColor,
            dark: matchMedia('(prefers-color-scheme: dark)').matches};
        });
        const levels = [luminance(actual.color), luminance(actual.background)];
        const contrast = (Math.max(...levels) + 0.05) / (Math.min(...levels) + 0.05);
        if (actual.dark !== (theme === 'dark') || actual.text !== fixture.text || contrast < 4.5) {
          throw new Error(JSON.stringify({scenario: fixture.id, theme, contrast, actual}));
        }
        cases++;
      }
    }
    console.log(JSON.stringify({cases}));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


class TestBrowser(unittest.TestCase):
    @unittest.skipUnless(
        NODE and PUPPETEER and render.chrome_available(),
        "optional Node/Puppeteer and Chrome not configured",
    )
    def test_baseline_values_and_contrast_in_both_themes(self):
        fixtures = []
        for scenario in schema.load_all_scenarios().values():
            presentation = interfaces.variant(scenario, "baseline")
            fixtures.append({
                "id": scenario.id,
                "html": presentation.to_html(),
                "text": interfaces.render_text(presentation, include_header=False),
            })
        result = subprocess.run(
            [NODE, "-e", SCRIPT, PUPPETEER, render.chrome_path()],
            input=json.dumps(fixtures), text=True, capture_output=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cases"], 6)
