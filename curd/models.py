from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

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


# 初始化数据库连接和会话
def init_db():
    # 正式站数据库配置
    DB_CONFIG = {
        "host": MysqlConfig.host,
        "user": MysqlConfig.user,
        "password": MysqlConfig.password,
        "database": MysqlConfig.database,
        "port": MysqlConfig.port,
        "charset": MysqlConfig.charset
    }

    # 创建数据库连接字符串
    db_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
             f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?" \
             f"charset={DB_CONFIG['charset']}"

    # 创建数据库引擎
    engine = create_engine(db_url, echo=False)

    # 创建会话类
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # 创建数据库表（如果不存在）
    Base.metadata.create_all(bind=engine)

    return SessionLocal
