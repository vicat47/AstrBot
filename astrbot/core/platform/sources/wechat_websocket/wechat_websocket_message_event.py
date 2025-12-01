import asyncio
from datetime import datetime
from typing import AsyncGenerator

import aiohttp

from astrbot import logger
from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform import AstrMessageEvent, AstrBotMessage, PlatformMetadata, MessageType
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
        async with aiohttp.ClientSession() as session:
            for comp in message.chain:
                await asyncio.sleep(1)
                if isinstance(comp, Plain):
                    await self._send_text(session, comp.text)

    async def _send_text(self, session: aiohttp.ClientSession, text: str):
        payload = {
            "para": {
                "id": f"{int(datetime.now().timestamp())}",
                "type": 555,
                "roomid": "null",
                "wxid": "null",
                "content": "1",
                "nickname": "null",
                "ext": "null"
            }
        }
        if (
            self.message_obj.type == MessageType.GROUP_MESSAGE  # 确保是群聊消息
            and self.adapter.settings.get(
                "reply_with_mention",
                False,
            )  # 检查适配器设置是否启用 reply_with_mention
            and self.message_obj.sender  # 确保有发送者信息
            and (
                self.message_obj.sender.user_id or self.message_obj.sender.nickname
            )  # 确保发送者有 ID 或昵称
        ):
            payload["para"]["roomid"] = self.message_obj.group.group_id
            mention_text = (
                    self.message_obj.sender.nickname or self.message_obj.sender.user_id
            )
            message_text = f"@{mention_text} {text}"
        else:
            payload["para"]["wxid"] = self.message_obj.sender.user_id
            message_text = text
        if self.get_group_id() and "#" in self.session_id:
            session_id = self.session_id.split("#")[0]
        else:
            session_id = self.session_id
        url = f"{self.adapter.base_url}/api/sendtxtmsg"
        await self._post(session, url, payload)

    async def _post(self, session, url, payload):
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if resp.status != 200 or data.get("Code") != 200:
                    logger.error(f"{url} failed: {resp.status} {data}")
        except Exception as e:
            logger.error(f"{url} error: {e}")