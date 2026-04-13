import pygame
import numpy as np
import sys
import time


WIN_W  = 900
WIN_H  = 680
FPS    = 60
TOTAL  = 5000

SCENARIOS = {
    "normal":  {"blocked": [],    "speed": 1.0, "panic": 1.0},
    "partial": {"blocked": [1,5], "speed": 0.80,"panic": 1.3},
    "panic":   {"blocked": [1,5], "speed": 0.60,"panic": 2.0},
}

# --- COLOURS ---
BG     = (15, 20, 15)
FIELD  = (34, 85, 30)
PANEL  = (22, 28, 22)
WHITE  = (220, 220, 210)
GREEN  = (30, 200, 120)
RED    = (210, 60,  40)
ORANGE = (210, 130, 30)
TEAL   = (30, 158, 117)

# --- 8 exits placed around the oval ---
def make_exits():
    cx, cy = 350, 290
    rx, ry = 280, 220
    angles = [90, 45, 0, 315, 270, 225, 180, 135]
    names  = ["N","NE","E","SE","S","SW","W","NW"]
    exits  = []
    for i, (a, n) in enumerate(zip(angles, names)):
        rad = np.radians(a)
        x   = int(cx + rx * np.cos(rad))
        y   = int(cy - ry * np.sin(rad))
        exits.append({"x": x, "y": y, "name": n, "id": i})
    return exits

EXITS = make_exits()


# --- ONE PERSON (agent) ---
class Person:
    def __init__(self, x, y, exit_id, speed):
        self.x       = float(x)
        self.y       = float(y)
        self.exit_id = exit_id
        self.gx      = float(EXITS[exit_id]["x"])
        self.gy      = float(EXITS[exit_id]["y"])
        self.speed   = speed
        self.out     = False
        self.color   = (50, 200, 100)

    def move(self, crowd):
        if self.out:
            return

        dx   = self.gx - self.x
        dy   = self.gy - self.y
        dist = np.hypot(dx, dy)

        if dist < 6:
            self.out = True
            return

        px, py = 0.0, 0.0
        me = id(self) % len(crowd)
        for k in range(10):
            nb  = crowd[(me + k * 91) % len(crowd)]
            if nb is self or nb.out:
                continue
            ddx = self.x - nb.x
            ddy = self.y - nb.y
            gap = np.hypot(ddx, ddy)
            if 0 < gap < 10:
                f   = (10 - gap) / 10 * 0.35
                px += ddx / gap * f
                py += ddy / gap * f

        spd    = self.speed * (0.5 + 0.5 * dist / (dist + 25))
        self.x += (dx / dist) * spd + px
        self.y += (dy / dist) * spd + py

        heat       = max(0, 1 - dist / 200)
        r          = int(50  + heat * 180)
        g          = int(200 - heat * 170)
        self.color = (r, g, 50)

    def draw(self, screen):
        if not self.out:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 2)


# --- THE SIMULATION ---
class Sim:
    def __init__(self, key):
        self.key        = key
        self.sc         = SCENARIOS[key]
        self.blocked    = self.sc["blocked"]
        self.crowd      = []
        self.tick       = 0
        self.done       = False
        self.ms         = {25:None, 50:None, 75:None, 90:None, 100:None}
        self.max_q      = np.zeros(8, dtype=int)
        self.start_time = None   # set on first step
        self.elapsed    = 0.0  
        self.place_people()

    def place_people(self):
        open_exits = [e for e in EXITS if e["id"] not in self.blocked]
        rng        = np.random.default_rng(42)
        cx, cy     = 350, 290

        for _ in range(TOTAL):
            r   = 0.45 + rng.random() * 0.48
            ang = rng.random() * 2 * np.pi
            x   = cx + r * 240 * np.cos(ang) + rng.normal(0, 8)
            y   = cy + r * 185 * np.sin(ang) + rng.normal(0, 8)

            nearest = min(open_exits, key=lambda e: np.hypot(e["x"]-x, e["y"]-y))
            spd     = (0.9 + rng.random() * 0.8) * self.sc["speed"]
            self.crowd.append(Person(x, y, nearest["id"], spd))

    def step(self):
        if self.done:
            return

        # start clockon fthe first step
        if self.start_time is None:
            self.start_time = time.time()

        self.elapsed = time.time() - self.start_time

        q = np.zeros(8, dtype=int)
        for p in self.crowd:
            p.move(self.crowd)
            if not p.out:
                d = np.hypot(EXITS[p.exit_id]["x"] - p.x, EXITS[p.exit_id]["y"] - p.y)
                if d < 40:
                    q[p.exit_id] += 1

        np.maximum(self.max_q, q, out=self.max_q)

        out = self.evacuated()
        pct = out / TOTAL * 100

        for m in [25, 50, 75, 90, 100]:
            if self.ms[m] is None and pct >= m:
                self.ms[m] = round(self.elapsed, 1)

        if out >= TOTAL:
            self.done       = True
            self.total_time = round(self.elapsed, 1)

        self.tick += 1

    def evacuated(self):
        return sum(1 for p in self.crowd if p.out)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Sikaville Stadium Evacuation")
        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        self.font    = pygame.font.SysFont("monospace", 11)
        self.bold    = pygame.font.SysFont("monospace", 13, bold=True)
        self.timer   = pygame.time.Clock()
        self.sim     = None
        self.key     = "normal"
        self.running = False
        self.done    = False
        self.build_arena()

    def build_arena(self):
        self.arena = pygame.Surface((700, 580))
        s = self.arena
        s.fill((25, 32, 25))
        pygame.draw.ellipse(s, BG,    (10, 10, 680, 560))
        pygame.draw.ellipse(s, FIELD, (245, 215, 210, 150))
        pygame.draw.ellipse(s, (50, 120, 45), (245, 215, 210, 150), 2)

    def draw_arena(self):
        self.screen.blit(self.arena, (0, 0))
        blocked = SCENARIOS[self.key]["blocked"]

        if self.sim:
            for p in self.sim.crowd:
                p.draw(self.screen)
            for e in EXITS:
                q = int(self.sim.max_q[e["id"]])
                if q > 5 and e["id"] not in blocked:
                    ring_r = min(45, 12 + q // 5)
                    pygame.draw.circle(self.screen, ORANGE, (e["x"], e["y"]), ring_r, 2)

        for e in EXITS:
            col = RED if e["id"] in blocked else GREEN
            pygame.draw.circle(self.screen, col, (e["x"], e["y"]), 11)
            lbl = self.font.render(e["name"], True, (0, 0, 0))
            self.screen.blit(lbl, (e["x"] - lbl.get_width()//2, e["y"] - 5))

    def write(self, text, x, y, color=WHITE, bold=False):
        f = self.bold if bold else self.font
        self.screen.blit(f.render(text, True, color), (x, y))

    def draw_panel(self):
        px = 705
        pygame.draw.rect(self.screen, PANEL, (px, 0, WIN_W - px, WIN_H))

        y = 10
        self.write("SIKAVILLE STADIUM", px+5, y, TEAL, bold=True); y += 22
        self.write("Evacuation Sim",    px+5, y); y += 20
        pygame.draw.line(self.screen, TEAL, (px, y), (WIN_W, y), 1); y += 8

        self.write("SCENARIO:", px+5, y, TEAL, bold=True); y += 16
        for k in SCENARIOS:
            active = (k == self.key)
            bg     = TEAL    if active else (38, 48, 38)
            tc     = (0,0,0) if active else WHITE
            pygame.draw.rect(self.screen, bg, (px+4, y, 182, 20), border_radius=3)
            self.write((">" if active else " ") + " " + k.upper(), px+8, y+3, tc)
            y += 24
        y += 4

        bw = 85
        pygame.draw.rect(self.screen, TEAL,   (px+4,    y, bw, 22), border_radius=3)
        pygame.draw.rect(self.screen, ORANGE, (px+8+bw, y, bw, 22), border_radius=3)
        self.write("RUN",   px+30,    y+4, (0,0,0), bold=True)
        self.write("RESET", px+14+bw, y+4, (0,0,0), bold=True)
        self._btn_run   = pygame.Rect(px+4,    y, bw, 22)
        self._btn_reset = pygame.Rect(px+8+bw, y, bw, 22)
        y += 30

        pygame.draw.line(self.screen, (50,60,50), (px, y), (WIN_W, y), 1); y += 8

        if self.sim:
            out = self.sim.evacuated()
            pct = out / TOTAL * 100
            t   = self.sim.elapsed          

            self.write("LIVE STATS", px+5, y, TEAL, bold=True); y += 16

            bar = 182
            pygame.draw.rect(self.screen, (40,50,40), (px+4, y, bar,              8), border_radius=3)
            pygame.draw.rect(self.screen, TEAL,       (px+4, y, int(bar*pct/100), 8), border_radius=3)
            y += 12

            self.write(f"Out  : {out:,} / {TOTAL:,}", px+5, y); y += 14
            self.write(f"Done : {pct:.1f}%",          px+5, y); y += 14
            self.write(f"Time : {t:.1f}s",            px+5, y); y += 18

            self.write("MILESTONES", px+5, y, TEAL, bold=True); y += 16
            for m in [25, 50, 75, 90, 100]:
                v = self.sim.ms[m]
                self.write(f"  {m}%  : {v if v else '---'}s", px+5, y); y += 13

            y += 6
            self.write("EXIT QUEUES", px+5, y, TEAL, bold=True); y += 16
            for e in EXITS:
                q   = int(self.sim.max_q[e["id"]])
                col = RED if q > 120 else ORANGE if q > 60 else WHITE
                self.write(f"  {e['name']:2s} : {q}", px+5, y, col); y += 13

        if self.done:
            y += 8
            t = self.sim.total_time
            self.write(f"Done in  {t}s total", px+5, y, GREEN, bold=True)

        self.write("SPACE=Run  R=Reset  ESC=Quit", px+4, WIN_H-16, (80,100,80))

    def click(self, pos):
        x, y = pos
        sy   = 66
        for k in SCENARIOS:
            if 705+4 <= x <= 705+186 and sy <= y <= sy+20:
                self.key     = k
                self.sim     = None
                self.running = False
                self.done    = False
            sy += 24

        if hasattr(self, "_btn_run") and self._btn_run.collidepoint(pos):
            if not self.running and not self.done:
                self.sim     = Sim(self.key)
                self.running = True

        if hasattr(self, "_btn_reset") and self._btn_reset.collidepoint(pos):
            self.sim     = None
            self.running = False
            self.done    = False

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if ev.key == pygame.K_r:
                        self.sim = None; self.running = False; self.done = False
                    if ev.key == pygame.K_SPACE and not self.running and not self.done:
                        self.sim = Sim(self.key); self.running = True
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    self.click(ev.pos)

            if self.running and self.sim:
                for _ in range(2):
                    self.sim.step()
                if self.sim.done:
                    self.running = False
                    self.done    = True

            self.screen.fill(BG)
            self.draw_arena()
            self.draw_panel()
            pygame.display.flip()
            self.timer.tick(FPS)


if __name__ == "__main__":
    App().run()