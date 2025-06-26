import re
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from curd.async_models import AITask, AITaskResult, init_db_async

from sqlalchemy import func, delete

from sqlalchemy.future import select


@asynccontextmanager
async def get_db_async():
    SessionLocal = init_db_async()
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_task_async(db: AsyncSession, task_id: str):
    result = await db.execute(select(AITask).where(AITask.task_id == task_id))
    return result.scalars().first()


# 创建任务
async def create_task_async(db: AsyncSession, task_data: dict):
    db_task = AITask(
        task_id=task_data["task_id"],
        status=task_data["status"],
        message=task_data["message"],
        progress=task_data["progress"],
        original_path=task_data["original_path"],
        created_at=datetime.now()
    )
    db.add(db_task)
    await db.commit()
    await db.refresh(db_task)
    return db_task


async def update_task_status_async(db: AsyncSession, task_data: dict):
    task_id = task_data["task_id"]
    result = await db.execute(select(AITask).where(AITask.task_id == task_id))
    db_task = result.scalars().first()
    if not db_task:
        return None

    # 更新字段
    db_task.status = task_data.get("status", db_task.status)
    db_task.message = task_data.get("message", db_task.message)
    db_task.progress = task_data.get("progress", db_task.progress)
    db_task.segments_path = task_data.get("segments_path", db_task.segments_path)
    db_task.start_time = task_data.get("start_time", db_task.start_time)
    db_task.complete_time = task_data.get("complete_time", db_task.complete_time)
    db_task.duration = task_data.get("duration", db_task.duration)
    db_task.error = task_data.get("error", db_task.error)

    # 如果状态完成或失败，设置完成时间
    if db_task.status in ["completed", "failed"] and not db_task.complete_time:
        db_task.complete_time = datetime.now()
        if db_task.start_time:
            duration = (db_task.complete_time - db_task.start_time).total_seconds()
            db_task.duration = round(duration, 2)

    await db.commit()
    await db.refresh(db_task)
    return db_task


async def get_task_results_async(
        db: AsyncSession,
        task_id: str,
        keyword: str = None,
        speaker: str = None,
        page: int = 1,
        per_page: int = 10
):
    stmt = select(AITaskResult).where(AITaskResult.task_id == task_id)

    if keyword:
        keywords = re.split(r'[;,；，、]', keyword)
        # keywords = [kw.strip() for kw in keyword.split(',') if kw.strip()]
        if keywords:
            keyword_conditions = [
                (AITaskResult.text.ilike(f"%{kw}%")) | (AITaskResult.speaker.ilike(f"%{kw}%"))
                for kw in keywords
            ]
            stmt = stmt.where(or_(*keyword_conditions))

    if speaker:
        stmt = stmt.where(AITaskResult.speaker.ilike(f"%{speaker}%"))

    # 获取总数
    count_stmt = select(func.count()).select_from(AITaskResult).where(AITaskResult.task_id == task_id)

    if keyword:
        count_keywords_conditions = [
            (AITaskResult.text.ilike(f"%{kw}%")) | (AITaskResult.speaker.ilike(f"%{kw}%"))
            for kw in keywords
        ]
        count_stmt = count_stmt.where(or_(*count_keywords_conditions))

    if speaker:
        count_stmt = count_stmt.where(AITaskResult.speaker.ilike(f"%{speaker}%"))

    result = await db.execute(count_stmt)
    total = result.scalar()

    # 分页查询
    stmt = stmt.order_by(AITaskResult.index).offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(stmt)
    items = result.scalars().all()

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 1
    }


async def get_all_task_results_async(db: AsyncSession, task_id: str):
    result = await db.execute(
        select(AITaskResult).where(AITaskResult.task_id == task_id)
    )
    return result.scalars().all()


# 删除任务
async def delete_task_async(db: AsyncSession, task_id: str):
    # 删除子表数据
    await db.execute(
        delete(AITaskResult).where(AITaskResult.task_id == task_id)
    )

    # 删除主表数据
    await db.execute(
        delete(AITask).where(AITask.task_id == task_id)
    )

    await db.commit()


async def get_segments_by_indices_async(db: AsyncSession, task_id: str, indices: list[int]):
    result = await db.execute(
        select(AITaskResult).where(
            AITaskResult.task_id == task_id,
            AITaskResult.index.in_(indices)
        )
    )
    return result.scalars().all()


async def get_all_segments_async(db: AsyncSession, task_id: str):
    result = await db.execute(
        select(AITaskResult).where(AITaskResult.task_id == task_id)
    )
    return result.scalars().all()


from sqlalchemy import or_


async def delete_segments_by_keywords_async(db: AsyncSession, task_id: str, keywords: list):
    stmt = select(AITaskResult).where(AITaskResult.task_id == task_id)

    conditions = []
    for k in keywords:
        conditions.append(
            (AITaskResult.text.like(k)) | (AITaskResult.speaker.like(k))
        )

    if conditions:
        stmt = stmt.where(or_(*conditions))

    result = await db.execute(stmt)
    results = result.scalars().all()

    if not results:
        return []

    deleted_indices = [seg.index for seg in results]

    delete_stmt = delete(AITaskResult).where(
        AITaskResult.task_id == task_id,
        AITaskResult.index.in_([seg.index for seg in results])
    )

    await db.execute(delete_stmt)
    await db.commit()

    return deleted_indices


async def delete_segments_by_indices_async(db: AsyncSession, task_id: str, indices: list):
    result = await db.execute(
        select(AITaskResult).where(
            AITaskResult.task_id == task_id,
            AITaskResult.index.in_(indices)
        )
    )
    results = result.scalars().all()

    if not results:
        return []

    deleted_indices = [seg.index for seg in results]

    delete_stmt = delete(AITaskResult).where(
        AITaskResult.task_id == task_id,
        AITaskResult.index.in_(deleted_indices)
    )

    await db.execute(delete_stmt)
    await db.commit()

    return deleted_indices

