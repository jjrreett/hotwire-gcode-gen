import asyncio
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import sys


async def ainput(string: str) -> str:
    await asyncio.to_thread(sys.stdout.write, f"{string} ")
    return (await asyncio.to_thread(sys.stdin.readline)).rstrip("\n")


class AsyncVisualizer:
    def __init__(self):
        self.plotter = BackgroundPlotter()
        self.queue = asyncio.Queue()
        self.state = {"radius": 1.0, "position": [0, 0, 0]}

        # Create an initial sphere
        self.sphere = pv.Sphere(
            radius=self.state["radius"], center=self.state["position"]
        )
        self.actor = self.plotter.add_mesh(self.sphere, color="blue")

    async def update_visualization(self):
        """Update the visualization based on new state changes"""
        while True:
            state_update = await self.queue.get()
            if state_update is None:  # Exit signal
                break

            # Update state
            self.state.update(state_update)

            # Modify the sphere
            self.plotter.remove_actor(self.actor)  # Remove old actor
            self.sphere = pv.Sphere(
                radius=self.state["radius"], center=self.state["position"]
            )
            self.actor = self.plotter.add_mesh(self.sphere, color="blue")

            self.plotter.update()  # Refresh the visualization
            print(f"Updated Visualization: {self.state}")

    async def cli_session(self):
        """CLI loop to take user input and update the queue"""
        print("CLI interactive session started. Type 'exit' to quit.")
        while True:
            user_input = await ainput("Enter 'radius <value>' or 'position x y z': ")
            user_input = user_input.strip()
            if user_input.lower() == "exit":
                await self.queue.put(None)
                break
            try:
                parts = user_input.split()
                if parts[0] == "radius" and len(parts) == 2:
                    radius = float(parts[1])
                    await self.queue.put({"radius": radius})
                elif parts[0] == "position" and len(parts) == 4:
                    position = list(map(float, parts[1:4]))
                    await self.queue.put({"position": position})
                else:
                    print("Invalid command. Try again.")
            except ValueError:
                print("Invalid input format. Try again.")


async def main():
    vis = AsyncVisualizer()

    # Run visualization and CLI in parallel
    await asyncio.gather(vis.update_visualization(), vis.cli_session())


if __name__ == "__main__":
    asyncio.run(main())
