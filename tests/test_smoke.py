from pathlib import Path
import ast, importlib, unittest
ROOT = Path(__file__).resolve().parents[1]
class SmokeTests(unittest.TestCase):
    def test_python_compiles(self):
        for directory in ['src','scripts']:
            for f in (ROOT/directory).rglob('*.py'):compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')

    def test_imports_and_payoff(self):
        from delta_hedge.pricing import option_price
        import delta_hedge.hedging, delta_hedge.simulation, delta_hedge.backtesting
        from delta_hedge.paths import DATA
        self.assertEqual(DATA,ROOT/'data/raw')
        self.assertEqual(option_price(120,100,0,0.05,0.2,'call'),20)
        import scripts.run_analysis

if __name__ == '__main__':unittest.main()
