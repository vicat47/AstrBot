from typing import AsyncGenerator

from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform import AstrMessageEvent, AstrBotMessage, PlatformMetadata
from astrbot.core.platform.sources.wechat_websocket.wechat_websocket_adapter import WeChatWebsocketAdapter

# TODO: 完善代码
class WeChatWebsocketMessageEvent(AstrMessageEvent):
    def __init__(self,
                 message_str: str,
                 message_obj: AstrBotMessage,
                 platform_meta: PlatformMetadata,
                 session_id: str,
                 adapter: "WeChatWebsocketAdapter", ):
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.message_obj = message_obj  # Save the full message object
        self.adapter = adapter

    async def send_streaming(self, generator: AsyncGenerator[MessageChain, None], use_fallback: bool = False):
        return await super().send_streaming(generator, use_fallback)

    async def send(self, message: MessageChain):
        return await super().send(message)