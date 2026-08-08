from datetime import timedelta

from temporalio import workflow

from workers.temporal_activities import free_collect_activity


@workflow.defn
class FreeCollectWorkflow:
    @workflow.run
    async def run(self) -> None:
        await workflow.execute_activity(
            free_collect_activity,
            start_to_close_timeout=timedelta(minutes=10),
        )
