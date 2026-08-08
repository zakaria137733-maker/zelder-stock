import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import seeding


async def main():
    print("Seeding databases...")
    credentials = await seeding.seed_customers()
    points = seeding.seed_influx()
    print(f"Seeded {len(credentials)} customers and {points} InfluxDB points")
    print("Its Done.")


if __name__ == "__main__":
    asyncio.run(main())
