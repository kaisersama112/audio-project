from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime, Float, ForeignKey, create_engine

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

from config import MysqlConfig

Base = declarative_base()


# 定义任务表模型
class AITask(Base):
    __tablename__ = 'ai_tasks'
    task_id = Column(String(36), primary_key=True)
    status = Column(String(20), nullable=False, default='pending')
    message = Column(Text)
    progress = Column(Integer, nullable=False, default=0)
    original_path = Column(Text)
    segments_path = Column(Text)
    start_time = Column(DateTime)
    complete_time = Column(DateTime)
    duration = Column(Float)
    is_upload = Column(Integer, nullable=False, default=0)
    error = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.now)


# 定义任务结果表模型
class AITaskResult(Base):
    __tablename__ = 'ai_task_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey('ai_tasks.task_id'), nullable=False)
    index = Column(Integer, nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    speaker = Column(String(100))
    url = Column(String(1000))
    task = relationship("AITask", backref="results")


# 定义下载任务表模型
class AIDownloadTask(Base):
    __tablename__ = 'ai_download_tasks'
    id = Column(Integer, primary_key=True, autoincrement=True)  # 使用自增整数作为主键
    task_id = Column(String(36), unique=True, nullable=False)  # 唯一的下载任务ID
    original_task_id = Column(String(36), nullable=False)  # 关联原始任务的
    status = Column(String(20), nullable=False, default='pending')
    progress = Column(Integer, nullable=False, default=0)
    file_url = Column(Text)
    download_path = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)



def init_db_async():
    """异步数据库引擎"""
    DB_CONFIG = {
        "host": MysqlConfig.host,
        "user": MysqlConfig.user,
        "password": MysqlConfig.password,
        "database": MysqlConfig.database,
        "port": MysqlConfig.port,
        "charset": MysqlConfig.charset
    }

    db_url = f"mysql+asyncmy://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"

    engine = create_async_engine(
        db_url,
        pool_size=10, # 连接池大小
        max_overflow=5, # 连接池溢出
        pool_recycle=30, # 连接池回收时间 （秒）
        pool_pre_ping=True,
        echo=False,
    )

    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    return SessionLocal
