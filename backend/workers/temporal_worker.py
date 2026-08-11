import asyncio
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
)
from temporalio.worker import Worker

from workers.temporal_activities import free_collect_activity, seed_demo_trades_activity
from workers.temporal_workflows import FreeCollectWorkflow, SeedDemoTradesWorkflow

TASK_QUEUE = "sentimentiq"


async def ensure_schedule(client, schedule_id, workflow_run, task_queue):
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_run,
            id=schedule_id,
            task_queue=task_queue,
        ),
        spec=ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=timedelta(minutes=5))],
        ),
    )
    try:
        await client.create_schedule(schedule_id, schedule)
        print(f"Schedule created: {schedule_id}")
    except Exception as e:
        print(f"Schedule {schedule_id} could not be created: {e}")


async def main() -> None:
    from config import settings

    client = await Client.connect(settings.temporal_url)

    await ensure_schedule(client, "collect-free-every-5min", FreeCollectWorkflow.run, TASK_QUEUE)
    await ensure_schedule(client, "seed-demo-trades-every-5min", SeedDemoTradesWorkflow.run, TASK_QUEUE)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FreeCollectWorkflow, SeedDemoTradesWorkflow],
        activities=[free_collect_activity, seed_demo_trades_activity],
    )
    print("Worker started on task queue:", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
