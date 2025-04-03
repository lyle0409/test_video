
from manim import *
from manim_voiceover import VoiceoverScene
from src.utils.kokoro_voiceover import KokoroService  # You MUST import like this as this is our custom voiceover service.

# Helper Functions/Classes
class Scene6_Helper:
    def __init__(self, scene):
        self.scene = scene

    def create_triangle(self):
        # Ensure vertices are 3D points
        triangle = Polygon([[0, 0, 0], [5, 0, 0], [0, 7, 0]], color=BLUE)
        triangle.move_to(ORIGIN)
        return triangle

    def create_label(self, text, position):
        return MathTex(text).set_color(WHITE).scale(0.75).move_to(position)

    def create_formula_tex(self, formula_str):
        return MathTex(formula_str).scale(0.75)

    def get_edge_centers(self, polygon, buff=0.3):
        # Get the vertices of the polygon
        vertices = polygon.get_vertices()
        n_vertices = len(vertices)
        edge_centers = []
        for i in range(n_vertices):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n_vertices]
            edge_center = (v1 + v2) / 2
            edge_centers.append(edge_center + np.array([0, -buff, 0]))  # Adjust for buff
        return edge_centers

class Scene6(VoiceoverScene):
    def construct(self):
        # Initialize speech service
        self.set_speech_service(KokoroService())

        # Instantiate helper class
        helper = Scene6_Helper(self)

        # --- Stage 1: Triangle Setup ---
        triangle = helper.create_triangle()
        self.play(Create(triangle), run_time=2)

        # Create labels for sides
        edge_centers = helper.get_edge_centers(triangle)
        a_label = helper.create_label("a = 5", edge_centers[0] + np.array([0, 0.3, 0]))
        b_label = helper.create_label("b = 7", edge_centers[1] + np.array([0.3, 0, 0]))
        c_label = helper.create_label("c", edge_centers[2] + np.array([0, -0.5, 0]))
        
        self.play(Write(a_label), Write(b_label), Write(c_label))

        # Create label for angle A
        A_label = helper.create_label("A = 30^{\\circ}", triangle.get_vertices()[0] + np.array([-0.5, 0.3, 0]))
        self.play(Write(A_label))

        # --- Stage 2: Introduce the Law of Sines ---
        formula_label = helper.create_formula_tex(r"\frac{a}{\sin(A)} = \frac{b}{\sin(B)}")
        formula_label.next_to(triangle, RIGHT, buff=0.5)
        self.play(Write(formula_label), run_time=2)

        # --- Stage 3: Calculate B ---
        B_calculation = helper.create_formula_tex(r"\sin(B) = \frac{b \cdot \sin(A)}{a}")
        B_calculation.next_to(formula_label, DOWN, buff=0.3)
        self.play(Write(B_calculation), run_time=2)

        # Create label for angle B
        B_label = helper.create_label("B", triangle.get_vertices()[1] + np.array([0, -0.5, 0]))
        self.play(Write(B_label), run_time=1)

        # --- Stage 4: Calculate c ---
        C_calculation = helper.create_formula_tex(r"c = \frac{a \cdot \sin(C)}{\sin(A)}")
        C_calculation.next_to(B_calculation, DOWN, buff=0.3)
        self.play(Write(C_calculation), run_time=2)

        # Create label for calculated side c
        c_calculated = helper.create_label("c \\approx 8.24", triangle.get_vertices()[2] + np.array([0, -0.5, 0]))
        self.play(Write(c_calculated), run_time=1)

        self.wait(1)  # Scene end transition buffer
