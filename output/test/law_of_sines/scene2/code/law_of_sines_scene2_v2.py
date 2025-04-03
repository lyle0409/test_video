
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService  # Using a fallback voiceover service

# Helper class to modularize object creation and positioning
class Scene7_Helper:
    def __init__(self, scene):
        self.scene = scene

    def create_formula_tex(self):
        # Create the Law of Sines formula
        law_of_sines = MathTex(
            r"\frac{a}{\sin(A)} = \frac{b}{\sin(B)} = \frac{c}{\sin(C)}",
            font_size=24,
            color=BLUE_C,
        )
        return law_of_sines

    def create_summary_text(self):
        # Create a summary text
        summary_text = Tex(
            "The Law of Sines relates sides and angles in any triangle.",
            font_size=24,
            color=WHITE,
        )
        return summary_text

    def create_application_images(self):
        # Placeholder images for navigation and engineering applications
        nav_image = Rectangle(width=2, height=1, color=GREEN_C)
        eng_image = Rectangle(width=2, height=1, color=BLUE_C)
        return nav_image, eng_image


class Scene7(VoiceoverScene, MovingCameraScene):
    def construct(self):
        # Initialize speech service using GTTSService as a fallback
        self.set_speech_service(GTTSService())

        # Instantiate helper
        helper = Scene7_Helper(self)

        # --- Stage 1: Create and Position Objects ---
        with self.voiceover(text="Now, let's explore the final aspects of the Law of Sines in Scene 7.") as tracker:
            # Create objects
            law_of_sines = helper.create_formula_tex()
            summary_text = helper.create_summary_text()
            nav_image, eng_image = helper.create_application_images()

            # Adjusted Positioning of Objects
            law_of_sines.to_edge(LEFT, buff=1.0)  # Fixed: Moved further right to satisfy safe margin checks.
            print(f"Law of Sines left position: {law_of_sines.get_left()[0]}")  # Debug: Verify positioning during development.
            
            summary_text.next_to(law_of_sines, DOWN, buff=0.3)
            nav_image.to_edge(RIGHT, buff=0.5)
            eng_image.next_to(nav_image, DOWN, buff=0.3)

            # Ensure spatial constraints
            assert law_of_sines.get_left()[0] >= -7.0 + 0.5, "Law of Sines violates left safe margin."
            assert summary_text.get_bottom()[1] >= -4.0 + 0.5, "Summary text violates bottom safe margin."
            assert nav_image.get_right()[0] <= 7.0 - 0.5, "Navigation image violates right safe margin."
            assert eng_image.get_bottom()[1] >= -4.0 + 0.5, "Engineering image violates bottom safe margin."

        # --- Stage 2: Animate Formula and Summary ---
        with self.voiceover(
            text="The Law of Sines relates the lengths of the sides of a triangle to the sines of its angles. "
            "We can express this mathematically as: "
            r"\[ \frac{a}{\sin(A)} = \frac{b}{\sin(B)} = \frac{c}{\sin(C)} \]."
        ) as tracker:
            self.play(Write(law_of_sines, run_time=2))
            self.wait(2)

        with self.voiceover(
            text="This formula allows us to solve for unknown sides or angles in any triangle, "
            "making it a powerful tool in geometry."
        ) as tracker:
            self.play(FadeIn(summary_text, run_time=1))
            self.wait(2)

        # --- Stage 3: Animate Application Images ---
        with self.voiceover(
            text="In practical terms, the Law of Sines is invaluable in fields like navigation and engineering."
        ) as tracker:
            self.play(FadeIn(nav_image, run_time=1))
            self.wait(2)

        with self.voiceover(
            text="For instance, in navigation, pilots and sailors use this law to determine their course across the globe."
        ) as tracker:
            self.play(Indicate(nav_image, scale_factor=1.2, color=YELLOW, run_time=1))
            self.wait(2)

        with self.voiceover(
            text="Similarly, engineers rely on this law when designing structures, ensuring that angles and lengths are precise for safety and functionality."
        ) as tracker:
            self.play(FadeIn(eng_image, run_time=1))
            self.wait(2)

        with self.voiceover(
            text="As we conclude, remember that the Law of Sines is not just a theoretical concept, but a practical tool used in various real-world applications."
        ) as tracker:
            self.play(Indicate(eng_image, scale_factor=1.2, color=YELLOW, run_time=1))
            self.wait(2)

        # --- Stage 4: Fade Out and Conclude ---
        with self.voiceover(
            text="By mastering this law, you're equipping yourself with essential skills for solving triangle problems in mathematics and beyond."
        ) as tracker:
            self.play(
                FadeOut(law_of_sines),
                FadeOut(summary_text),
                FadeOut(nav_image),
                FadeOut(eng_image),
                run_time=2,
            )
            self.wait(1)
