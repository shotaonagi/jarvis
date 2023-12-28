"""Initial migration

Revision ID: 2913fddf7e71
Revises: 
Create Date: 2023-12-28 14:52:02.999365

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2913fddf7e71'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('assistants',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('model', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_assistants_id'), 'assistants', ['id'], unique=False)
    op.create_index(op.f('ix_assistants_name'), 'assistants', ['name'], unique=False)
    op.create_table('files',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('file_content', sa.LargeBinary(), nullable=True),
    sa.Column('purpose', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_files_id'), 'files', ['id'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(), nullable=True),
    sa.Column('email', sa.String(), nullable=True),
    sa.Column('hashed_password', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('threads',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('assistant_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['assistant_id'], ['assistants.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_threads_id'), 'threads', ['id'], unique=False)
    op.create_table('messages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('thread_id', sa.Integer(), nullable=True),
    sa.Column('content', sa.String(), nullable=True),
    sa.Column('role', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['thread_id'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_messages_id'), table_name='messages')
    op.drop_table('messages')
    op.drop_index(op.f('ix_threads_id'), table_name='threads')
    op.drop_table('threads')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_files_id'), table_name='files')
    op.drop_table('files')
    op.drop_index(op.f('ix_assistants_name'), table_name='assistants')
    op.drop_index(op.f('ix_assistants_id'), table_name='assistants')
    op.drop_table('assistants')
    # ### end Alembic commands ###