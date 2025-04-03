
from manim import *
from manim_voiceover import VoiceoverScene
from src.utils.kokoro_voiceover import KokoroService  # You MUST import like this as this is our custom voiceover service.

class Scene4_Helper:  
    def __init__(self, scene):
        self.scene = scene

    def create_triangle(self, vertices):
        return Polygon(*vertices)

    def create_angle_label(self, text, triangle, position):
        angle_label = MathTex(text, color=YELLOW).scale(0.75)
        angle_label.next_to(triangle, position, buff=0.3)  # Maintain minimum spacing
        return angle_label

    def create_law_of_sines_label(self):
        return MathTex(r"\frac{a}{\sin A} = \frac{b}{\sin B}", color=BLUE_C).scale(0.75)

class Scene4(VoiceoverScene, MovingCameraScene):  
    def construct(self):
        # Initialize speech service
        self.set_speech_service(KokoroService())

        # Instantiate helper class
        helper = Scene4_Helper(self)

        # --- Stage 1: Create Triangles ---
        with self.voiceover(text="Now, let's explore how we can derive the Law of Sines from the properties of similar triangles.") as tracker:
            # Create the first triangle (left)
            triangle_A = helper.create_triangle([(-2, 0, 0), (-2, 2, 0), (-1, 0, 0)])
            # Create the second triangle (right)
            triangle_B = helper.create_triangle([(0, 0, 0), (0, 2, 0), (1, 0, 0)])

            # Create dashed line for height
            dashed_line = Line(start=(-2, 2, 0), end=(-2, 0, 0), dashed=True)

            # Group triangles for simultaneous creation
            triangle_group = VGroup(triangle_A, triangle_B)

            # Animate triangles and dashed line creation
            self.play(Create(triangle_group), run_time=2)
            self.wait(1)  # Allow viewers to absorb the triangles
            self.play(Create(dashed_line), run_time=1)
            self.wait(1)  # Allow viewers to absorb the height representation

        # --- Stage 2: Add Angle Labels ---
        with self.voiceover(text="Here we have two right-angled triangles, Triangle A and Triangle B, which share a common angle, denoted as \\( \\angle A \\). Notice how these triangles are positioned side by side.") as tracker:
            angle_A = helper.create_angle_label(r"\angle A", triangle_A, UP)
            angle_B = helper.create_angle_label(r"\angle B", triangle_B, UP)

            self.play(Write(angle_A), run_time=1)
            self.play(Write(angle_B), run_time=1)
            self.wait(1)  # Allow viewers to absorb the angles

        # --- Stage 3: Law of Sines ---
        with self.voiceover(text="Finally, we can express this relationship mathematically with the Law of Sines: \\( \\frac{a}{\\sin A} = \\frac{b}{\\sin B} \\).") as tracker:
            law_of_sines = helper.create_law_of_sines_label()
            law_of_sines.move_to(DOWN * 2)  # Position below the triangles

            self.play(Write(law_of_sines), run_time=1.5)
            self.wait(1)  # Final pause for viewers to absorb the concept

        self.wait(1)  # Scene end transition buffer
